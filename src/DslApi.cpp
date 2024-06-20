/*
The MIT License

Copyright (c) 2019-2024, Prominence AI, Inc.

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
#include "DslServices.h"
#include "spdlog/spdlog.h"

#define RETURN_IF_PARAM_IS_NULL(input_string) do \
{ \
    if (!input_string) \
    { \
        LOG_ERROR("Input parameter must be a valid address - not NULL"); \
        return DSL_RESULT_INVALID_INPUT_PARAM; \
    } \
}while(0); 

#define RETURN_IF_NEW_NVSTREAMMUX_ENABLED() do \
{ \
    if (DSL::Services::GetServices()->UseNewStreammuxGet()) \
    { \
        LOG_ERROR("USE_NEW_NVSTREAMMUX must NOT be set to 'yes' to enabled service"); \
        return DSL_RESULT_API_NOT_ENABLED; \
    } \
}while(0); 

#define RETURN_IF_NEW_NVSTREAMMUX_DISABLED() do \
{ \
    if (!DSL::Services::GetServices()->UseNewStreammuxGet()) \
    { \
        LOG_ERROR("USE_NEW_NVSTREAMMUX must be set to 'yes' to enabled service"); \
        return DSL_RESULT_API_NOT_ENABLED; \
    } \
}while(0); 

DslReturnType dsl_display_type_rgba_color_custom_new(const wchar_t* name, 
    double red, double green, double blue, double alpha)
{
    RETURN_IF_PARAM_IS_NULL(name);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaColorNew(
        cstrName.c_str(), red, green, blue, alpha);
}

DslReturnType dsl_display_type_rgba_color_predefined_new(const wchar_t* name, 
    uint color_id, double alpha)
{
    RETURN_IF_PARAM_IS_NULL(name);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaColorPredefinedNew(
        cstrName.c_str(), color_id, alpha);
}

DslReturnType dsl_display_type_rgba_color_random_new(const wchar_t* name, 
    uint hue, uint luminosity, double alpha, uint seed)
{
    RETURN_IF_PARAM_IS_NULL(name);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaColorRandomNew(
        cstrName.c_str(), hue, luminosity, alpha, seed);
}

DslReturnType dsl_display_type_rgba_color_on_demand_new(const wchar_t* name, 
    dsl_display_type_rgba_color_provider_cb provider, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaColorOnDemandNew(
        cstrName.c_str(), provider, client_data);
}

DslReturnType dsl_display_type_rgba_color_palette_new(const wchar_t* name, 
    const wchar_t** colors)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(colors);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    std::vector<std::shared_ptr<std::string>> newColors; 
    std::vector<const char*> cColors;
    
    for (const wchar_t** color = colors; *color; color++)
    {
        std::wstring wstrColor(*color);
        std::string cstrColor(wstrColor.begin(), wstrColor.end());
        
        std::cout << "new color = " << cstrColor.c_str() << "\n";
        
        std::shared_ptr<std::string> newColor = 
            std::shared_ptr<std::string>(new std::string(cstrColor.c_str()));
        newColors.push_back(newColor);
        cColors.push_back(newColor->c_str());
    }
    cColors.push_back(NULL);

    return DSL::Services::GetServices()->DisplayTypeRgbaColorPaletteNew(
        cstrName.c_str(), &cColors[0], newColors.size());
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_display_type_rgba_color_palette_predefined_new(const wchar_t* name, 
    uint palette_id, double alpha)
{    
    RETURN_IF_PARAM_IS_NULL(name);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaColorPalettePredefinedNew(
        cstrName.c_str(), palette_id, alpha);
}

DslReturnType dsl_display_type_rgba_color_palette_random_new(const wchar_t* name, 
    uint size, uint hue, uint luminosity, double alpha, uint seed)
{
    RETURN_IF_PARAM_IS_NULL(name);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaColorPaletteRandomNew(
        cstrName.c_str(), size, hue, luminosity, alpha, seed);
}

DslReturnType dsl_display_type_rgba_color_palette_index_get(const wchar_t* name, 
    uint* index)
{
    RETURN_IF_PARAM_IS_NULL(name);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaColorPaletteIndexGet(
        cstrName.c_str(), index);
}
    
DslReturnType dsl_display_type_rgba_color_palette_index_set(const wchar_t* name, 
    uint index)
{
    RETURN_IF_PARAM_IS_NULL(name);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaColorPaletteIndexSet(
        cstrName.c_str(), index);
}
    
DslReturnType dsl_display_type_rgba_color_next_set(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaColorNextSet(
        cstrName.c_str());
}
    
DslReturnType dsl_display_type_rgba_font_new(const wchar_t* name, 
    const wchar_t* font, uint size, const wchar_t* color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(font);
    RETURN_IF_PARAM_IS_NULL(color);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFont(font);
    std::string cstrFont(wstrFont.begin(), wstrFont.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaFontNew(
        cstrName.c_str(), cstrFont.c_str(), size, cstrColor.c_str());
}

DslReturnType dsl_display_type_rgba_text_new(const wchar_t* name, 
    const wchar_t* text, uint x_offset, uint y_offset, const wchar_t* font, 
    boolean has_bg_color, const wchar_t* bg_color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(text);
    RETURN_IF_PARAM_IS_NULL(font);

    std::string cstrBgColor;
    if (has_bg_color)
    {
        RETURN_IF_PARAM_IS_NULL(bg_color);
        std::wstring wstrBgColor(bg_color);
        cstrBgColor.assign(wstrBgColor.begin(), wstrBgColor.end());
    }

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrText(text);
    std::string cstrText(wstrText.begin(), wstrText.end());
    std::wstring wstrFont(font);
    std::string cstrFont(wstrFont.begin(), wstrFont.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaTextNew(
        cstrName.c_str(), cstrText.c_str(), x_offset, y_offset, 
        cstrFont.c_str(), has_bg_color, cstrBgColor.c_str());
}

DslReturnType dsl_display_type_rgba_line_new(const wchar_t* name, 
    uint x1, uint y1, uint x2, uint y2, uint width, const wchar_t* color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(color);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaLineNew(cstrName.c_str(), 
        x1, y1, x2, y2, width, cstrColor.c_str());
}
    
DslReturnType dsl_display_type_rgba_arrow_new(const wchar_t* name, 
    uint x1, uint y1, uint x2, uint y2, uint width, uint head, 
    const wchar_t* color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(color);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaArrowNew(
        cstrName.c_str(), x1, y1, x2, y2, width, head, cstrColor.c_str());
}
    
DslReturnType dsl_display_type_rgba_rectangle_new(const wchar_t* name, 
    uint left, uint top, uint width, uint height, uint border_width, 
    const wchar_t* color, bool has_bg_color, const wchar_t* bg_color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(color);

    std::string cstrBgColor;
    if (has_bg_color)
    {
        RETURN_IF_PARAM_IS_NULL(bg_color);
        std::wstring wstrBgColor(bg_color);
        cstrBgColor.assign(wstrBgColor.begin(), wstrBgColor.end());
    }

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaRectangleNew(
        cstrName.c_str(), left, top, width, height, border_width, 
        cstrColor.c_str(), has_bg_color, cstrBgColor.c_str());
}

DslReturnType dsl_display_type_rgba_polygon_new(const wchar_t* name, 
    const dsl_coordinate* coordinates, uint num_coordinates, 
    uint border_width, const wchar_t* color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(coordinates);
    RETURN_IF_PARAM_IS_NULL(color);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaPolygonNew(
        cstrName.c_str(), coordinates, num_coordinates, border_width, 
        cstrColor.c_str());
}
    
DslReturnType dsl_display_type_rgba_line_multi_new(const wchar_t* name, 
    const dsl_coordinate* coordinates, uint num_coordinates, 
    uint border_width, const wchar_t* color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(coordinates);
    RETURN_IF_PARAM_IS_NULL(color);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaLineMultiNew(
        cstrName.c_str(), coordinates, num_coordinates, border_width, 
        cstrColor.c_str());
}
    
DslReturnType dsl_display_type_rgba_circle_new(const wchar_t* name, 
    uint x_center, uint y_center, uint radius, const wchar_t* color, 
    bool has_bg_color, const wchar_t* bg_color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(color);

    std::string cstrBgColor;
    if (has_bg_color)
    {
        RETURN_IF_PARAM_IS_NULL(bg_color);
        std::wstring wstrBgColor(bg_color);
        cstrBgColor.assign(wstrBgColor.begin(), wstrBgColor.end());
    }

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaCircleNew(
        cstrName.c_str(), x_center, y_center, radius, cstrColor.c_str(), 
        has_bg_color, cstrBgColor.c_str());
}

DslReturnType dsl_display_type_source_unique_id_new(const wchar_t* name, 
    uint x_offset, uint y_offset, const wchar_t* font, boolean has_bg_color, 
    const wchar_t* bg_color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(font);

    std::string cstrBgColor;
    if (has_bg_color)
    {
        RETURN_IF_PARAM_IS_NULL(bg_color);
        std::wstring wstrBgColor(bg_color);
        cstrBgColor.assign(wstrBgColor.begin(), wstrBgColor.end());
    }

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFont(font);
    std::string cstrFont(wstrFont.begin(), wstrFont.end());

    return DSL::Services::GetServices()->DisplayTypeSourceUniqueIdNew(
        cstrName.c_str(), x_offset, y_offset, cstrFont.c_str(), has_bg_color, 
        cstrBgColor.c_str());
}

DslReturnType dsl_display_type_source_stream_id_new(const wchar_t* name, 
    uint x_offset, uint y_offset, const wchar_t* font, boolean has_bg_color, 
    const wchar_t* bg_color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(font);

    std::string cstrBgColor;
    if (has_bg_color)
    {
        RETURN_IF_PARAM_IS_NULL(bg_color);
        std::wstring wstrBgColor(bg_color);
        cstrBgColor.assign(wstrBgColor.begin(), wstrBgColor.end());
    }

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFont(font);
    std::string cstrFont(wstrFont.begin(), wstrFont.end());

    return DSL::Services::GetServices()->DisplayTypeSourceStreamIdNew(
        cstrName.c_str(), x_offset, y_offset, cstrFont.c_str(), has_bg_color, 
        cstrBgColor.c_str());
}

DslReturnType dsl_display_type_source_name_new(const wchar_t* name, 
    uint x_offset, uint y_offset, const wchar_t* font, boolean has_bg_color, 
    const wchar_t* bg_color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(font);

    std::string cstrBgColor;
    if (has_bg_color)
    {
        RETURN_IF_PARAM_IS_NULL(bg_color);
        std::wstring wstrBgColor(bg_color);
        cstrBgColor.assign(wstrBgColor.begin(), wstrBgColor.end());
    }

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFont(font);
    std::string cstrFont(wstrFont.begin(), wstrFont.end());

    return DSL::Services::GetServices()->DisplayTypeSourceNameNew(cstrName.c_str(),
        x_offset, y_offset, cstrFont.c_str(), has_bg_color, cstrBgColor.c_str());
}

DslReturnType dsl_display_type_source_dimensions_new(const wchar_t* name, 
    uint x_offset, uint y_offset, const wchar_t* font, boolean has_bg_color, 
    const wchar_t* bg_color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(font);

    std::string cstrBgColor;
    if (has_bg_color)
    {
        RETURN_IF_PARAM_IS_NULL(bg_color);
        std::wstring wstrBgColor(bg_color);
        cstrBgColor.assign(wstrBgColor.begin(), wstrBgColor.end());
    }

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFont(font);
    std::string cstrFont(wstrFont.begin(), wstrFont.end());

    return DSL::Services::GetServices()->DisplayTypeSourceDimensionsNew(
        cstrName.c_str(), x_offset, y_offset, cstrFont.c_str(), has_bg_color, 
        cstrBgColor.c_str());
}

DslReturnType dsl_display_type_source_frame_rate_new(const wchar_t* name, 
    uint x_offset, uint y_offset, const wchar_t* font, boolean has_bg_color, 
    const wchar_t* bg_color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(font);

    std::string cstrBgColor;
    if (has_bg_color)
    {
        RETURN_IF_PARAM_IS_NULL(bg_color);
        std::wstring wstrBgColor(bg_color);
        cstrBgColor.assign(wstrBgColor.begin(), wstrBgColor.end());
    }

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFont(font);
    std::string cstrFont(wstrFont.begin(), wstrFont.end());

    return DSL::Services::GetServices()->DisplayTypeSourceFrameRateNew(
        cstrName.c_str(), x_offset, y_offset, cstrFont.c_str(), has_bg_color, 
        cstrBgColor.c_str());
}

DslReturnType dsl_display_type_rgba_text_shadow_add(const wchar_t* name, 
    uint x_offset, uint y_offset, const wchar_t* color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(color);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());

    return DSL::Services::GetServices()->DisplayRgbaTextShadowAdd(
        cstrName.c_str(), x_offset, y_offset, cstrColor.c_str());
}
    
DslReturnType dsl_display_type_meta_add(const wchar_t* name, 
    void* display_meta, void* frame_meta)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTypeMetaAdd(cstrName.c_str(), 
        display_meta, frame_meta);
}
    
DslReturnType dsl_display_type_delete(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTypeDelete(cstrName.c_str());
}

DslReturnType dsl_display_type_delete_many(const wchar_t** names)
{
    RETURN_IF_PARAM_IS_NULL(names);

    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());

        DslReturnType retval = DSL::Services::GetServices()->DisplayTypeDelete(cstrName.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_display_type_delete_all()
{
    return DSL::Services::GetServices()->DisplayTypeDeleteAll();
}

uint dsl_display_type_list_size()
{
    return DSL::Services::GetServices()->DisplayTypeListSize();
}

DslReturnType dsl_ode_action_bbox_format_new(const wchar_t* name, uint border_width, 
    const wchar_t* border_color, boolean has_bg_color, const wchar_t* bg_color)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    std::string cstrBorderColor;
    if (border_width)
    {
        RETURN_IF_PARAM_IS_NULL(border_color);
        std::wstring wstrBorderColor(border_color);
        cstrBorderColor.assign(wstrBorderColor.begin(), wstrBorderColor.end());
    }
    
    std::string cstrBgColor;
    if (has_bg_color)
    {
        RETURN_IF_PARAM_IS_NULL(bg_color);
        std::wstring wstrBgColor(bg_color);
        cstrBgColor.assign(wstrBgColor.begin(), wstrBgColor.end());
    }
    
    return DSL::Services::GetServices()->OdeActionBBoxFormatNew(cstrName.c_str(), 
        border_width, cstrBorderColor.c_str(), has_bg_color, cstrBgColor.c_str());
}

DslReturnType dsl_ode_action_bbox_scale_new(const wchar_t* name, uint scale)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionBBoxScaleNew(cstrName.c_str(), 
        scale);
}

DslReturnType dsl_ode_action_bbox_style_corners_new(const wchar_t* name, 
    const wchar_t* color, uint length, uint max_length,
    dsl_threshold_value* thickness_values, uint num_values)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(color);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());
    
    return DSL::Services::GetServices()->OdeActionBBoxStyleCornersNew(
        cstrName.c_str(), cstrColor.c_str(), length, max_length,
        thickness_values, num_values);
}
    
DslReturnType dsl_ode_action_bbox_style_crosshair_new(const wchar_t* name, 
    const wchar_t* color, uint radius, uint max_radius, uint inner_radius,
    dsl_threshold_value* thickness_values, uint num_values)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(color);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());
    
    return DSL::Services::GetServices()->OdeActionBBoxStyleCrosshairNew(
        cstrName.c_str(), cstrColor.c_str(), radius, max_radius, inner_radius,
        thickness_values, num_values);
}
    
DslReturnType dsl_ode_action_label_format_new(const wchar_t* name, 
    const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    std::string cstrFont;
    if (font != NULL)
    {
        std::wstring wstrFont(font);
        cstrFont.assign(wstrFont.begin(), wstrFont.end());
    }
    
    std::string cstrBgColor;
    if (has_bg_color)
    {
        RETURN_IF_PARAM_IS_NULL(bg_color);
        std::wstring wstrBgColor(bg_color);
        cstrBgColor.assign(wstrBgColor.begin(), wstrBgColor.end());
    }
    
    return DSL::Services::GetServices()->OdeActionLabelFormatNew(cstrName.c_str(), 
        cstrFont.c_str(), has_bg_color, cstrBgColor.c_str());
}
    
DslReturnType dsl_ode_action_custom_new(const wchar_t* name, 
    dsl_ode_handle_occurrence_cb client_hanlder, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(client_hanlder);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionCustomNew(cstrName.c_str(), 
        client_hanlder, client_data);
}

DslReturnType dsl_ode_action_capture_frame_new(const wchar_t* name, 
    const wchar_t* outdir)
{
#if !defined(BUILD_WITH_FFMPEG) || !defined(BUILD_WITH_OPENCV)
    #error "BUILD_WITH_FFMPEG and BUILD_WITH_OPENCV must be defined"
#elif (BUILD_WITH_FFMPEG != true) && (BUILD_WITH_OPENCV != true)
    LOG_ERROR("dsl_sink_frame_capture_new requires one of BUILD_WITH_FFMPEG \
       or BUILD_WITH_OPENCV to be set true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else    
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(outdir);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOutdir(outdir);
    std::string cstrOutdir(wstrOutdir.begin(), wstrOutdir.end());

    return DSL::Services::GetServices()->OdeActionCaptureFrameNew(cstrName.c_str(), 
        cstrOutdir.c_str());
#endif        
}

DslReturnType dsl_ode_action_capture_object_new(const wchar_t* name,
    const wchar_t* outdir)
{
#if !defined(BUILD_WITH_FFMPEG) || !defined(BUILD_WITH_OPENCV)
    #error "BUILD_WITH_FFMPEG and BUILD_WITH_OPENCV must be defined"
#elif (BUILD_WITH_FFMPEG != true) && (BUILD_WITH_OPENCV != true)
    LOG_ERROR("dsl_sink_frame_capture_new requires one of BUILD_WITH_FFMPEG \
       or BUILD_WITH_OPENCV to be set true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else    
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(outdir);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOutdir(outdir);
    std::string cstrOutdir(wstrOutdir.begin(), wstrOutdir.end());

    return DSL::Services::GetServices()->OdeActionCaptureObjectNew(cstrName.c_str(), 
        cstrOutdir.c_str());
#endif
}

DslReturnType dsl_ode_action_capture_complete_listener_add(const wchar_t* name, 
    dsl_capture_complete_listener_cb listener, void* client_data)
{
#if !defined(BUILD_WITH_FFMPEG) || !defined(BUILD_WITH_OPENCV)
    #error "BUILD_WITH_FFMPEG and BUILD_WITH_OPENCV must be defined"
#elif (BUILD_WITH_FFMPEG != true) && (BUILD_WITH_OPENCV != true)
    LOG_ERROR("dsl_sink_frame_capture_new requires one of BUILD_WITH_FFMPEG \
       or BUILD_WITH_OPENCV to be set true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else    
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        OdeActionCaptureCompleteListenerAdd(cstrName.c_str(), listener, client_data);
#endif
}
    
DslReturnType dsl_ode_action_capture_complete_listener_remove(const wchar_t* name, 
    dsl_capture_complete_listener_cb listener)
{
#if !defined(BUILD_WITH_FFMPEG) || !defined(BUILD_WITH_OPENCV)
    #error "BUILD_WITH_FFMPEG and BUILD_WITH_OPENCV must be defined"
#elif (BUILD_WITH_FFMPEG != true) && (BUILD_WITH_OPENCV != true)
    LOG_ERROR("dsl_sink_frame_capture_new requires one of BUILD_WITH_FFMPEG \
       or BUILD_WITH_OPENCV to be set true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else    
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        OdeActionCaptureCompleteListenerRemove(cstrName.c_str(), listener);
#endif
}
    
DslReturnType dsl_ode_action_capture_image_player_add(const wchar_t* name, 
    const wchar_t* player)
{
#if !defined(BUILD_WITH_FFMPEG) || !defined(BUILD_WITH_OPENCV)
    #error "BUILD_WITH_FFMPEG and BUILD_WITH_OPENCV must be defined"
#elif (BUILD_WITH_FFMPEG != true) && (BUILD_WITH_OPENCV != true)
    LOG_ERROR("dsl_sink_frame_capture_new requires one of BUILD_WITH_FFMPEG \
       or BUILD_WITH_OPENCV to be set true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else    
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(player);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPlayer(player);
    std::string cstrPlayer(wstrPlayer.begin(), wstrPlayer.end());

    return DSL::Services::GetServices()->
        OdeActionCaptureImagePlayerAdd(cstrName.c_str(), cstrPlayer.c_str());
#endif
}
    
DslReturnType dsl_ode_action_capture_image_player_remove(const wchar_t* name, 
    const wchar_t* player)
{
#if !defined(BUILD_WITH_FFMPEG) || !defined(BUILD_WITH_OPENCV)
    #error "BUILD_WITH_FFMPEG and BUILD_WITH_OPENCV must be defined"
#elif (BUILD_WITH_FFMPEG != true) && (BUILD_WITH_OPENCV != true)
    LOG_ERROR("dsl_sink_frame_capture_new requires one of BUILD_WITH_FFMPEG \
       or BUILD_WITH_OPENCV to be set true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else    
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(player);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPlayer(player);
    std::string cstrPlayer(wstrPlayer.begin(), wstrPlayer.end());

    return DSL::Services::GetServices()->
        OdeActionCaptureImagePlayerRemove(cstrName.c_str(), cstrPlayer.c_str());
#endif
}
    
DslReturnType dsl_ode_action_capture_mailer_add(const wchar_t* name, 
    const wchar_t* mailer, const wchar_t* subject, boolean attach)
{
#if !defined(BUILD_WITH_FFMPEG) || !defined(BUILD_WITH_OPENCV)
    #error "BUILD_WITH_FFMPEG and BUILD_WITH_OPENCV must be defined"
#elif (BUILD_WITH_FFMPEG != true) && (BUILD_WITH_OPENCV != true)
    LOG_ERROR("dsl_sink_frame_capture_new requires one of BUILD_WITH_FFMPEG \
       or BUILD_WITH_OPENCV to be set true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else    
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(mailer);
    RETURN_IF_PARAM_IS_NULL(subject);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrMailer(mailer);
    std::string cstrMailer(wstrMailer.begin(), wstrMailer.end());
    std::wstring wstrSubject(subject);
    std::string cstrSubject(wstrSubject.begin(), wstrSubject.end());

    return DSL::Services::GetServices()->OdeActionCaptureMailerAdd(
        cstrName.c_str(), cstrMailer.c_str(), cstrSubject.c_str(), attach);
#endif
}
    
DslReturnType dsl_ode_action_capture_mailer_remove(const wchar_t* name, 
    const wchar_t* mailer)
{
#if !defined(BUILD_WITH_FFMPEG) || !defined(BUILD_WITH_OPENCV)
    #error "BUILD_WITH_FFMPEG and BUILD_WITH_OPENCV must be defined"
#elif (BUILD_WITH_FFMPEG != true) && (BUILD_WITH_OPENCV != true)
    LOG_ERROR("dsl_sink_frame_capture_new requires one of BUILD_WITH_FFMPEG \
       or BUILD_WITH_OPENCV to be set true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else    
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(mailer);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrMailer(mailer);
    std::string cstrMailer(wstrMailer.begin(), wstrMailer.end());

    return DSL::Services::GetServices()->OdeActionCaptureMailerRemove(
        cstrName.c_str(), cstrMailer.c_str());
#endif
}

DslReturnType dsl_ode_action_label_customize_new(const wchar_t* name,  
    const uint* content_types, uint size)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(content_types);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionLabelCustomizeNew(
        cstrName.c_str(), content_types, size);
}    

DslReturnType dsl_ode_action_label_customize_get(const wchar_t* name,  
    uint* content_types, uint* size)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(content_types);
    RETURN_IF_PARAM_IS_NULL(size);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionLabelCustomizeGet(
        cstrName.c_str(), content_types, size);
}    
    
DslReturnType dsl_ode_action_label_customize_set(const wchar_t* name,  
    const uint* content_types, uint size)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionLabelCustomizeSet(
        cstrName.c_str(), content_types, size);
}    

DslReturnType dsl_ode_action_label_offset_new(const wchar_t* name,  
    int offset_x, int offset_y)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionLabelOffsetNew(
        cstrName.c_str(), offset_x, offset_y);
}    
    
DslReturnType dsl_ode_action_label_snap_to_grid_new(const wchar_t* name,  
    uint module_width, uint module_height)
    
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionLabelSnapToGridNew(
        cstrName.c_str(), module_width, module_height);
}    

DslReturnType dsl_ode_action_label_connect_to_bbox_new(const wchar_t* name,  
    const wchar_t* line_color, uint line_width, uint bbox_point)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(line_color);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrLineColor(line_color);
    std::string cstrLineColor(wstrLineColor.begin(), wstrLineColor.end());

    return DSL::Services::GetServices()->OdeActionLabelConnectToBBoxNew(
        cstrName.c_str(), cstrLineColor.c_str(), line_width, bbox_point);
}      

DslReturnType dsl_ode_action_display_new(const wchar_t* name, 
    const wchar_t* format_string, uint offset_x, uint offset_y, 
    const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(format_string);
    RETURN_IF_PARAM_IS_NULL(font);

    std::string cstrBgColor;
    if (has_bg_color)
    {
        RETURN_IF_PARAM_IS_NULL(bg_color);
        std::wstring wstrBgColor(bg_color);
        cstrBgColor.assign(wstrBgColor.begin(), wstrBgColor.end());
    }

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFormatString(format_string);
    std::string cstrFormatString(wstrFormatString.begin(), wstrFormatString.end());
    std::wstring wstrFont(font);
    std::string cstrFont(wstrFont.begin(), wstrFont.end());

    return DSL::Services::GetServices()->OdeActionDisplayNew(cstrName.c_str(),
        cstrFormatString.c_str(), offset_x, offset_y, cstrFont.c_str(), 
        has_bg_color, cstrBgColor.c_str());
}

DslReturnType dsl_ode_action_handler_disable_new(const wchar_t* name, const wchar_t* handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());

    return DSL::Services::GetServices()->OdeActionHandlerDisableNew(cstrName.c_str(), 
        cstrHandler.c_str());
}

DslReturnType dsl_ode_action_email_new(const wchar_t* name, 
    const wchar_t* mailer, const wchar_t* subject)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(mailer);
    RETURN_IF_PARAM_IS_NULL(subject);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrMailer(mailer);
    std::string cstrMailer(wstrMailer.begin(), wstrMailer.end());
    std::wstring wstrSubject(subject);
    std::string cstrSubject(wstrSubject.begin(), wstrSubject.end());

    return DSL::Services::GetServices()->OdeActionEmailNew(cstrName.c_str(),
        cstrMailer.c_str(), cstrSubject.c_str());
}

DslReturnType dsl_ode_action_fill_surroundings_new(const wchar_t* name, const wchar_t* color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(color);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());

    return DSL::Services::GetServices()->OdeActionFillSurroundingsNew(cstrName.c_str(),
        cstrColor.c_str());
}

DslReturnType dsl_ode_action_fill_frame_new(const wchar_t* name, const wchar_t* color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(color);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());

    return DSL::Services::GetServices()->OdeActionFillFrameNew(cstrName.c_str(),
        cstrColor.c_str());
}

DslReturnType dsl_ode_action_log_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionLogNew(cstrName.c_str());
}

DslReturnType dsl_ode_action_message_meta_add_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionMessageMetaAddNew(cstrName.c_str());
}

DslReturnType dsl_ode_action_message_meta_type_get(const wchar_t* name,
    uint* meta_type)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(meta_type);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionMessageMetaTypeGet(
        cstrName.c_str(), meta_type);
}
    
DslReturnType dsl_ode_action_message_meta_type_set(const wchar_t* name,
    uint meta_type)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionMessageMetaTypeSet(
        cstrName.c_str(), meta_type);
}
   
DslReturnType dsl_ode_action_display_meta_add_new(const wchar_t* name, const wchar_t* display_type)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(display_type);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrType(display_type);
    std::string cstrType(wstrType.begin(), wstrType.end());

    return DSL::Services::GetServices()->OdeActionDisplayMetaAddNew(cstrName.c_str(), 
        cstrType.c_str());
}

DslReturnType dsl_ode_action_display_meta_add_many_new(const wchar_t* name, const wchar_t** display_types)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(display_types);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrType(*display_types);
    std::string cstrType(wstrType.begin(), wstrType.end());

    uint retval = DSL::Services::GetServices()->OdeActionDisplayMetaAddNew(cstrName.c_str(), 
        cstrType.c_str());

    for (const wchar_t** display_type = display_types; *display_type; display_type++)
    {
        wstrType.assign(*display_type);
        cstrType.assign(wstrType.begin(), wstrType.end());
        
        DslReturnType retval = DSL::Services::GetServices()->
            OdeActionDisplayMetaAddDisplayType(cstrName.c_str(), cstrType.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
        
}

DslReturnType dsl_ode_action_file_new(const wchar_t* name, 
    const wchar_t* file_path, uint mode, uint format, boolean force_flush)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->OdeActionFileNew(cstrName.c_str(),
        cstrFilePath.c_str(), mode, format, force_flush);
}

DslReturnType dsl_ode_action_monitor_new(const wchar_t* name, 
    dsl_ode_monitor_occurrence_cb client_monitor, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(client_monitor);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionMonitorNew(cstrName.c_str(),
        client_monitor, client_data);
}

DslReturnType dsl_ode_action_object_remove_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionObjectRemoveNew(cstrName.c_str());
}

DslReturnType dsl_ode_action_pipeline_pause_new(const wchar_t* name, 
    const wchar_t* pipeline)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->OdeActionPipelinePauseNew(cstrName.c_str(), 
        cstrPipeline.c_str());
}

DslReturnType dsl_ode_action_pipeline_play_new(const wchar_t* name, 
    const wchar_t* pipeline)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->OdeActionPipelinePlayNew(cstrName.c_str(), 
        cstrPipeline.c_str());
}

DslReturnType dsl_ode_action_pipeline_stop_new(const wchar_t* name, 
    const wchar_t* pipeline)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->OdeActionPipelineStopNew(cstrName.c_str(), 
        cstrPipeline.c_str());
}

DslReturnType dsl_ode_action_player_pause_new(const wchar_t* name, 
    const wchar_t* player)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(player);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPlayer(player);
    std::string cstrPlayer(wstrPlayer.begin(), wstrPlayer.end());

    return DSL::Services::GetServices()->OdeActionPlayerPauseNew(cstrName.c_str(), 
        cstrPlayer.c_str());
}

DslReturnType dsl_ode_action_player_play_new(const wchar_t* name, 
    const wchar_t* player)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(player);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPlayer(player);
    std::string cstrPlayer(wstrPlayer.begin(), wstrPlayer.end());

    return DSL::Services::GetServices()->OdeActionPlayerPlayNew(cstrName.c_str(), 
        cstrPlayer.c_str());
}

DslReturnType dsl_ode_action_player_stop_new(const wchar_t* name, 
    const wchar_t* player)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(player);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPlayer(player);
    std::string cstrPlayer(wstrPlayer.begin(), wstrPlayer.end());

    return DSL::Services::GetServices()->OdeActionPlayerStopNew(cstrName.c_str(), 
        cstrPlayer.c_str());
}

DslReturnType dsl_ode_action_print_new(const wchar_t* name, 
    boolean force_flush)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionPrintNew(cstrName.c_str(),
        force_flush);
}

DslReturnType dsl_ode_action_redact_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionRedactNew(cstrName.c_str());
}

DslReturnType dsl_ode_action_sink_add_new(const wchar_t* name,
    const wchar_t* pipeline, const wchar_t* sink)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(sink);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrSink(sink);
    std::string cstrSink(wstrSink.begin(), wstrSink.end());
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->OdeActionSinkAddNew(cstrName.c_str(),
        cstrPipeline.c_str(), cstrSink.c_str());
}

DslReturnType dsl_ode_action_sink_remove_new(const wchar_t* name,
    const wchar_t* pipeline, const wchar_t* sink)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(sink);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrSink(sink);
    std::string cstrSink(wstrSink.begin(), wstrSink.end());
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->OdeActionSinkRemoveNew(cstrName.c_str(),
        cstrPipeline.c_str(), cstrSink.c_str());
}

DslReturnType dsl_ode_action_sink_record_start_new(const wchar_t* name,
    const wchar_t* record_sink, uint start, uint duration, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(record_sink);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrSink(record_sink);
    std::string cstrSink(wstrSink.begin(), wstrSink.end());

    return DSL::Services::GetServices()->OdeActionSinkRecordStartNew(cstrName.c_str(), 
        cstrSink.c_str(), start, duration, client_data);
}

DslReturnType dsl_ode_action_sink_record_stop_new(const wchar_t* name,
    const wchar_t* record_sink)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(record_sink);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrSink(record_sink);
    std::string cstrSink(wstrSink.begin(), wstrSink.end());

    return DSL::Services::GetServices()->OdeActionSinkRecordStopNew(cstrName.c_str(), 
        cstrSink.c_str());
}

DslReturnType dsl_ode_action_source_add_new(const wchar_t* name,
    const wchar_t* pipeline, const wchar_t* source)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(source);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrSource(source);
    std::string cstrSource(wstrSource.begin(), wstrSource.end());
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->OdeActionSourceAddNew(cstrName.c_str(),
        cstrPipeline.c_str(), cstrSource.c_str());
}

DslReturnType dsl_ode_action_source_remove_new(const wchar_t* name,
    const wchar_t* pipeline, const wchar_t* source)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(source);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrSource(source);
    std::string cstrSource(wstrSource.begin(), wstrSource.end());
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->OdeActionSourceRemoveNew(cstrName.c_str(),
        cstrPipeline.c_str(), cstrSource.c_str());
}

DslReturnType dsl_ode_action_tap_record_start_new(const wchar_t* name,
    const wchar_t* record_tap, uint start, uint duration, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(record_tap);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrTap(record_tap);
    std::string cstrTap(wstrTap.begin(), wstrTap.end());

    return DSL::Services::GetServices()->OdeActionTapRecordStartNew(cstrName.c_str(), 
        cstrTap.c_str(), start, duration, client_data);
}

DslReturnType dsl_ode_action_tap_record_stop_new(const wchar_t* name,
    const wchar_t* record_tap)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(record_tap);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrTap(record_tap);
    std::string cstrTap(wstrTap.begin(), wstrTap.end());

    return DSL::Services::GetServices()->OdeActionTapRecordStopNew(cstrName.c_str(), 
        cstrTap.c_str());
}

DslReturnType dsl_ode_action_area_add_new(const wchar_t* name,
    const wchar_t* trigger, const wchar_t* area)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(trigger);
    RETURN_IF_PARAM_IS_NULL(area);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrArea(area);
    std::string cstrArea(wstrArea.begin(), wstrArea.end());
    std::wstring wstrTrigger(trigger);
    std::string cstrTrigger(wstrTrigger.begin(), wstrTrigger.end());

    return DSL::Services::GetServices()->OdeActionAreaAddNew(cstrName.c_str(),
        cstrTrigger.c_str(), cstrArea.c_str());
}

DslReturnType dsl_ode_action_area_remove_new(const wchar_t* name,
    const wchar_t* trigger, const wchar_t* area)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(trigger);
    RETURN_IF_PARAM_IS_NULL(area);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrArea(area);
    std::string cstrArea(wstrArea.begin(), wstrArea.end());
    std::wstring wstrTrigger(trigger);
    std::string cstrTrigger(wstrTrigger.begin(), wstrTrigger.end());

    return DSL::Services::GetServices()->OdeActionAreaRemoveNew(cstrName.c_str(),
        cstrTrigger.c_str(), cstrArea.c_str());
}

DslReturnType dsl_ode_action_trigger_reset_new(const wchar_t* name, const wchar_t* trigger)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(trigger);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrTrigger(trigger);
    std::string cstrTrigger(wstrTrigger.begin(), wstrTrigger.end());

    return DSL::Services::GetServices()->OdeActionTriggerResetNew(cstrName.c_str(),
        cstrTrigger.c_str());
}

DslReturnType dsl_ode_action_trigger_disable_new(const wchar_t* name, const wchar_t* trigger)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(trigger);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrTrigger(trigger);
    std::string cstrTrigger(wstrTrigger.begin(), wstrTrigger.end());

    return DSL::Services::GetServices()->OdeActionTriggerDisableNew(cstrName.c_str(),
        cstrTrigger.c_str());
}

DslReturnType dsl_ode_action_trigger_enable_new(const wchar_t* name, const wchar_t* trigger)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(trigger);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrTrigger(trigger);
    std::string cstrTrigger(wstrTrigger.begin(), wstrTrigger.end());

    return DSL::Services::GetServices()->OdeActionTriggerEnableNew(cstrName.c_str(),
        cstrTrigger.c_str());
}

DslReturnType dsl_ode_action_action_disable_new(const wchar_t* name, const wchar_t* action)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(action);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAction(action);
    std::string cstrAction(wstrAction.begin(), wstrAction.end());

    return DSL::Services::GetServices()->OdeActionActionDisableNew(cstrName.c_str(),
        cstrAction.c_str());
}

DslReturnType dsl_ode_action_action_enable_new(const wchar_t* name, const wchar_t* action)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(action);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAction(action);
    std::string cstrAction(wstrAction.begin(), wstrAction.end());

    return DSL::Services::GetServices()->OdeActionActionEnableNew(cstrName.c_str(),
        cstrAction.c_str());
}

DslReturnType dsl_ode_action_tiler_source_show_new(const wchar_t* name, 
    const wchar_t* tiler, uint timeout, boolean has_precedence)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(tiler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrTiler(tiler);
    std::string cstrTiler(wstrTiler.begin(), wstrTiler.end());

    return DSL::Services::GetServices()->OdeActionTilerShowSourceNew(cstrName.c_str(),
        cstrTiler.c_str(), timeout, has_precedence);
}

DslReturnType dsl_ode_action_branch_add_new(const wchar_t* name,
    const wchar_t* tee, const wchar_t* branch)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(tee);
    RETURN_IF_PARAM_IS_NULL(branch);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrTee(tee);
    std::string cstrTee(wstrTee.begin(), wstrTee.end());
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());

    return DSL::Services::GetServices()->OdeActionBranchAddNew(cstrName.c_str(),
        cstrTee.c_str(), cstrBranch.c_str());
}

DslReturnType dsl_ode_action_branch_add_to_new(const wchar_t* name,
    const wchar_t* demuxer, const wchar_t* branch)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(demuxer);
    RETURN_IF_PARAM_IS_NULL(branch);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrDemuxer(demuxer);
    std::string cstrDemuxer(wstrDemuxer.begin(), wstrDemuxer.end());
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());

    return DSL::Services::GetServices()->OdeActionBranchAddToNew(cstrName.c_str(),
        cstrDemuxer.c_str(), cstrBranch.c_str());
}

DslReturnType dsl_ode_action_branch_move_to_new(const wchar_t* name,
    const wchar_t* demuxer, const wchar_t* branch)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(demuxer);
    RETURN_IF_PARAM_IS_NULL(branch);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrDemuxer(demuxer);
    std::string cstrDemuxer(wstrDemuxer.begin(), wstrDemuxer.end());
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());

    return DSL::Services::GetServices()->OdeActionBranchMoveToNew(cstrName.c_str(),
        cstrDemuxer.c_str(), cstrBranch.c_str());
}
    
DslReturnType dsl_ode_action_branch_remove_new(const wchar_t* name,
    const wchar_t* tee, const wchar_t* branch)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(tee);
    RETURN_IF_PARAM_IS_NULL(branch);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrTee(tee);
    std::string cstrTee(wstrTee.begin(), wstrTee.end());
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());

    return DSL::Services::GetServices()->OdeActionBranchRemoveNew(cstrName.c_str(),
        cstrTee.c_str(), cstrBranch.c_str());
}
    
DslReturnType dsl_ode_action_enabled_get(const wchar_t* name, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionEnabledGet(cstrName.c_str(), enabled);
}

DslReturnType dsl_ode_action_enabled_set(const wchar_t* name, boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionEnabledSet(cstrName.c_str(), enabled);
}

DslReturnType dsl_ode_action_enabled_state_change_listener_add(const wchar_t* name,
    dsl_ode_enabled_state_change_listener_cb listener, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionEnabledStateChangeListenerAdd(
        cstrName.c_str(), listener, client_data);
}

DslReturnType dsl_ode_action_enabled_state_change_listener_remove(const wchar_t* name,
    dsl_ode_enabled_state_change_listener_cb listener)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionEnabledStateChangeListenerRemove(
        cstrName.c_str(), listener);
}
    
DslReturnType dsl_ode_action_delete(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionDelete(cstrName.c_str());
}

DslReturnType dsl_ode_action_delete_many(const wchar_t** names)
{
    RETURN_IF_PARAM_IS_NULL(names);

    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());

        DslReturnType retval = DSL::Services::GetServices()->OdeActionDelete(cstrName.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_ode_action_delete_all()
{
    return DSL::Services::GetServices()->OdeActionDeleteAll();
}

uint dsl_ode_action_list_size()
{
    return DSL::Services::GetServices()->OdeActionListSize();
}

DslReturnType dsl_ode_area_inclusion_new(const wchar_t* name, 
    const wchar_t* polygon, boolean show, uint bbox_test_point)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(polygon);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPolygon(polygon);
    std::string cstrPolygon(wstrPolygon.begin(), wstrPolygon.end());

    return DSL::Services::GetServices()->OdeAreaInclusionNew(cstrName.c_str(), 
        cstrPolygon.c_str(), show, bbox_test_point);
}

DslReturnType dsl_ode_area_exclusion_new(const wchar_t* name, 
    const wchar_t* polygon, boolean show, uint bbox_test_point)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(polygon);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPolygon(polygon);
    std::string cstrPolygon(wstrPolygon.begin(), wstrPolygon.end());

    return DSL::Services::GetServices()->OdeAreaExclusionNew(cstrName.c_str(), 
        cstrPolygon.c_str(), show, bbox_test_point);
}

DslReturnType dsl_ode_area_line_new(const wchar_t* name, 
    const wchar_t* line, boolean show, uint bbox_test_point)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(line);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrLine(line);
    std::string cstrLine(wstrLine.begin(), wstrLine.end());

    return DSL::Services::GetServices()->OdeAreaLineNew(cstrName.c_str(), 
        cstrLine.c_str(), show, bbox_test_point);
}

DslReturnType dsl_ode_area_line_multi_new(const wchar_t* name, 
    const wchar_t* multi_line, boolean show, uint bbox_test_point)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(multi_line);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrMultiLine(multi_line);
    std::string cstrMultiLine(wstrMultiLine.begin(), wstrMultiLine.end());

    return DSL::Services::GetServices()->OdeAreaLineMultiNew(cstrName.c_str(), 
        cstrMultiLine.c_str(), show, bbox_test_point);
}

DslReturnType dsl_ode_area_delete(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeAreaDelete(cstrName.c_str());
}

DslReturnType dsl_ode_area_delete_many(const wchar_t** names)
{
    RETURN_IF_PARAM_IS_NULL(names);

    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());

        DslReturnType retval = DSL::Services::GetServices()->OdeAreaDelete(cstrName.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_ode_area_delete_all()
{
    return DSL::Services::GetServices()->OdeAreaDeleteAll();
}

uint dsl_ode_area_list_size()
{
    return DSL::Services::GetServices()->OdeAreaListSize();
}

DslReturnType dsl_ode_trigger_always_new(const wchar_t* name, const wchar_t* source, uint when)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerAlwaysNew(cstrName.c_str(), cstrSource.c_str(), when);
}

DslReturnType dsl_ode_trigger_occurrence_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerOccurrenceNew(cstrName.c_str(), 
        cstrSource.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_absence_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerAbsenceNew(cstrName.c_str(), 
        cstrSource.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_instance_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerInstanceNew(cstrName.c_str(), 
        cstrSource.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_instance_count_settings_get(const wchar_t* name,
    uint* instance_count, uint* suppression_count)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(instance_count);
    RETURN_IF_PARAM_IS_NULL(suppression_count);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerInstanceCountSettingsGet(
        cstrName.c_str(), instance_count, suppression_count);
}
    
DslReturnType dsl_ode_trigger_instance_count_settings_set(const wchar_t* name,
    uint instance_count, uint suppression_count)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerInstanceCountSettingsSet(
        cstrName.c_str(), instance_count, suppression_count);
}
    
DslReturnType dsl_ode_trigger_intersection_new(const wchar_t* name, 
    const wchar_t* source, uint class_id_a, uint class_id_b, uint limit)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerIntersectionNew(cstrName.c_str(), 
        cstrSource.c_str(), class_id_a, class_id_b, limit);
}

DslReturnType dsl_ode_trigger_summation_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerSummationNew(cstrName.c_str(), 
        cstrSource.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_new_high_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit, uint preset)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerNewHighNew(cstrName.c_str(), 
        cstrSource.c_str(), class_id, limit, preset);
}

DslReturnType dsl_ode_trigger_new_low_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit, uint preset)    
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerNewLowNew(cstrName.c_str(), 
        cstrSource.c_str(), class_id, limit, preset);
}

    
DslReturnType dsl_ode_trigger_custom_new(const wchar_t* name, const wchar_t* source, 
    uint class_id, uint limit, dsl_ode_check_for_occurrence_cb client_checker, 
    dsl_ode_post_process_frame_cb client_post_processor, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(client_checker);
    RETURN_IF_PARAM_IS_NULL(client_post_processor);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerCustomNew(cstrName.c_str(), cstrSource.c_str(), 
        class_id, limit, client_checker, client_post_processor, client_data);
}
    
DslReturnType dsl_ode_trigger_count_new(const wchar_t* name, const wchar_t* source, 
    uint class_id, uint limit, uint minimum, uint maximum)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerCountNew(cstrName.c_str(), cstrSource.c_str(), 
        class_id, limit, minimum, maximum);
}

DslReturnType dsl_ode_trigger_count_range_get(const wchar_t* name, 
    uint* minimum, uint* maximum)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(minimum);
    RETURN_IF_PARAM_IS_NULL(maximum);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerCountRangeGet(cstrName.c_str(), 
        minimum, maximum);
}
    
DslReturnType dsl_ode_trigger_count_range_set(const wchar_t* name, 
    uint minimum, uint maximum)
{
    RETURN_IF_PARAM_IS_NULL(name);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerCountRangeSet(cstrName.c_str(), 
        minimum, maximum);
}

DslReturnType dsl_ode_trigger_distance_new(const wchar_t* name, const wchar_t* source, 
    uint class_id_a, uint class_id_b, uint limit, uint minimum, uint maximum, 
    uint test_point, uint test_method)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerDistanceNew(cstrName.c_str(), cstrSource.c_str(), 
        class_id_a, class_id_b, limit, minimum, maximum, test_point, test_method);
}

DslReturnType dsl_ode_trigger_distance_range_get(const wchar_t* name, 
    uint* minimum, uint* maximum)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(minimum);
    RETURN_IF_PARAM_IS_NULL(maximum);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDistanceRangeGet(
        cstrName.c_str(), minimum, maximum);
}
    
DslReturnType dsl_ode_trigger_distance_range_set(const wchar_t* name, 
    uint minimum, uint maximum)
{
    RETURN_IF_PARAM_IS_NULL(name);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDistanceRangeSet(
        cstrName.c_str(), minimum, maximum);
}

DslReturnType dsl_ode_trigger_distance_test_params_get(const wchar_t* name, 
    uint* test_point, uint* test_method)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(test_point);
    RETURN_IF_PARAM_IS_NULL(test_method);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDistanceTestParamsGet(
        cstrName.c_str(), test_point, test_method);
}

DslReturnType dsl_ode_trigger_distance_test_params_set(const wchar_t* name, 
    uint test_point, uint test_method)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDistanceTestParamsSet(
        cstrName.c_str(), test_point, test_method);
}
    
DslReturnType dsl_ode_trigger_smallest_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerSmallestNew(
        cstrName.c_str(), cstrSource.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_largest_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerLargestNew(
        cstrName.c_str(), cstrSource.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_cross_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit, uint min_frame_count, 
    uint max_frame_count, uint test_method)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerCrossNew(cstrName.c_str(), 
        cstrSource.c_str(), class_id, limit, min_frame_count, 
        max_frame_count, test_method);
}

DslReturnType dsl_ode_trigger_persistence_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit, uint minimum, uint maximum)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerPersistenceNew(
        cstrName.c_str(), cstrSource.c_str(), class_id, limit, minimum, maximum);
}

DslReturnType dsl_ode_trigger_persistence_range_get(const wchar_t* name, 
    uint* minimum, uint* maximum)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerPersistenceRangeGet(
        cstrName.c_str(), minimum, maximum);
}
    
DslReturnType dsl_ode_trigger_persistence_range_set(const wchar_t* name, 
    uint minimum, uint maximum)
{
    RETURN_IF_PARAM_IS_NULL(name);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerPersistenceRangeSet(
        cstrName.c_str(), minimum, maximum);
}
    
DslReturnType dsl_ode_trigger_latest_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerLatestNew(
        cstrName.c_str(), cstrSource.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_earliest_new(const wchar_t* name, 
    const wchar_t* source, uint class_id, uint limit)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerEarliestNew(
        cstrName.c_str(), cstrSource.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_cross_test_settings_get(const wchar_t* name, 
    uint* min_frame_count, uint* max_frame_count, uint* test_method)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerCrossTestSettingsGet(
        cstrName.c_str(), min_frame_count, max_frame_count, test_method);
}

DslReturnType dsl_ode_trigger_cross_test_settings_set(const wchar_t* name, 
    uint min_frame_count, uint max_frame_count, uint test_method)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerCrossTestSettingsSet(
        cstrName.c_str(), min_frame_count, max_frame_count, test_method);
}
    
DslReturnType dsl_ode_trigger_cross_view_settings_get(const wchar_t* name, 
    boolean* enabled, const wchar_t** color, uint* line_width)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);
    RETURN_IF_PARAM_IS_NULL(color);
    RETURN_IF_PARAM_IS_NULL(line_width);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* ccolor;
    static std::string cstrColor;
    static std::wstring wcstrColor;
    
    uint retval = DSL::Services::GetServices()->OdeTriggerCrossViewSettingsGet(
        cstrName.c_str(), enabled, &ccolor, line_width);

    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrColor.assign(ccolor);
        wcstrColor.assign(cstrColor.begin(), cstrColor.end());
        
        *color = wcstrColor.c_str();
    }
    return retval;
}

DslReturnType dsl_ode_trigger_cross_view_settings_set(const wchar_t* name, 
    boolean enabled, const wchar_t* color, uint line_width)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(color);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());

    return DSL::Services::GetServices()->OdeTriggerCrossViewSettingsSet(cstrName.c_str(), 
        enabled, cstrColor.c_str(), line_width);
}
    
DslReturnType dsl_ode_trigger_reset(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerReset(cstrName.c_str());
}

DslReturnType dsl_ode_trigger_reset_timeout_get(const wchar_t* name, uint *timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerResetTimeoutGet(
        cstrName.c_str(), timeout);
}

DslReturnType dsl_ode_trigger_reset_timeout_set(const wchar_t* name, uint timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerResetTimeoutSet(
        cstrName.c_str(), timeout);
}

DslReturnType dsl_ode_trigger_limit_state_change_listener_add(const wchar_t* name,
    dsl_ode_trigger_limit_state_change_listener_cb listener, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerLimitStateChangeListenerAdd(
        cstrName.c_str(), listener, client_data);
}

DslReturnType dsl_ode_trigger_limit_state_change_listener_remove(const wchar_t* name,
    dsl_ode_trigger_limit_state_change_listener_cb listener)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerLimitStateChangeListenerRemove(
        cstrName.c_str(), listener);
}
    
DslReturnType dsl_ode_trigger_enabled_get(const wchar_t* name, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerEnabledGet(cstrName.c_str(), enabled);
}

DslReturnType dsl_ode_trigger_enabled_set(const wchar_t* name, boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerEnabledSet(cstrName.c_str(), enabled);
}

DslReturnType dsl_ode_trigger_enabled_state_change_listener_add(const wchar_t* name,
    dsl_ode_enabled_state_change_listener_cb listener, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerEnabledStateChangeListenerAdd(
        cstrName.c_str(), listener, client_data);
}

DslReturnType dsl_ode_trigger_enabled_state_change_listener_remove(const wchar_t* name,
    dsl_ode_enabled_state_change_listener_cb listener)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerEnabledStateChangeListenerRemove(
        cstrName.c_str(), listener);
}
    
DslReturnType dsl_ode_trigger_class_id_get(const wchar_t* name, uint* class_id)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerClassIdGet(cstrName.c_str(), class_id);
}

DslReturnType dsl_ode_trigger_class_id_set(const wchar_t* name, uint class_id)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerClassIdSet(cstrName.c_str(), class_id);
}

DslReturnType dsl_ode_trigger_class_id_ab_get(const wchar_t* name, 
    uint* class_id_a, uint* class_id_b)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerClassIdABGet(cstrName.c_str(), 
        class_id_a, class_id_b);
}

DslReturnType dsl_ode_trigger_class_id_ab_set(const wchar_t* name, 
    uint class_id_a, uint class_id_b)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerClassIdABSet(cstrName.c_str(), 
        class_id_a, class_id_b);
}

DslReturnType dsl_ode_trigger_limit_event_get(const wchar_t* name, uint* limit)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(limit);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerLimitEventGet(cstrName.c_str(),
        limit);
}

DslReturnType dsl_ode_trigger_limit_event_set(const wchar_t* name, uint limit)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerLimitEventSet(cstrName.c_str(),
        limit);
}

DslReturnType dsl_ode_trigger_limit_frame_get(const wchar_t* name, uint* limit)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(limit);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerLimitFrameGet(cstrName.c_str(),
        limit);
}

DslReturnType dsl_ode_trigger_limit_frame_set(const wchar_t* name, uint limit)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerLimitFrameSet(cstrName.c_str(),
        limit);
}

DslReturnType dsl_ode_trigger_source_get(const wchar_t* name, const wchar_t** source)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(source);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cSource(NULL);
    static std::string cstrSource;
    static std::wstring wcstrSource;
    
    uint retval = DSL::Services::GetServices()->OdeTriggerSourceGet(cstrName.c_str(), &cSource);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        *source = NULL;
        if (cSource)
        {
            cstrSource.assign(cSource);
            wcstrSource.assign(cstrSource.begin(), cstrSource.end());
            *source = wcstrSource.c_str();
        }
    }
    return retval;

}

DslReturnType dsl_ode_trigger_source_set(const wchar_t* name, const wchar_t* source)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrSource;
    if (source)
    {
        std::wstring wstrSource(source);
        cstrSource.assign(wstrSource.begin(), wstrSource.end());
    }
    return DSL::Services::GetServices()->OdeTriggerSourceSet(cstrName.c_str(), cstrSource.c_str());
}

DslReturnType dsl_ode_trigger_infer_get(const wchar_t* name, const wchar_t** infer)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(infer);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cInfer(NULL);
    static std::string cstrInfer;
    static std::wstring wcstrInfer;
    
    uint retval = DSL::Services::GetServices()->OdeTriggerInferGet(cstrName.c_str(), 
        &cInfer);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        *infer = NULL;
        if (cInfer)
        {
            cstrInfer.assign(cInfer);
            wcstrInfer.assign(cstrInfer.begin(), cstrInfer.end());
            *infer = wcstrInfer.c_str();
        }
    }
    return retval;

}

DslReturnType dsl_ode_trigger_infer_set(const wchar_t* name, 
    const wchar_t* infer)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrInfer;
    if (infer)
    {
        std::wstring wstrInfer(infer);
        cstrInfer.assign(wstrInfer.begin(), wstrInfer.end());
    }
    return DSL::Services::GetServices()->OdeTriggerInferSet(
        cstrName.c_str(), cstrInfer.c_str());
}

DslReturnType dsl_ode_trigger_infer_confidence_min_get(const wchar_t* name, 
    float* min_confidence)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerConfidenceMinGet(
        cstrName.c_str(), min_confidence);
}

DslReturnType dsl_ode_trigger_infer_confidence_min_set(const wchar_t* name, 
    float min_confidence)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerConfidenceMinSet(
        cstrName.c_str(), min_confidence);
}

DslReturnType dsl_ode_trigger_infer_confidence_max_get(const wchar_t* name, 
    float* max_confidence)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerConfidenceMaxGet(
        cstrName.c_str(), max_confidence);
}

DslReturnType dsl_ode_trigger_infer_confidence_max_set(const wchar_t* name, 
    float max_confidence)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerConfidenceMaxSet(
        cstrName.c_str(), max_confidence);
}

DslReturnType dsl_ode_trigger_tracker_confidence_min_get(const wchar_t* name, 
    float* min_confidence)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerTrackerConfidenceMinGet(
        cstrName.c_str(), min_confidence);
}

DslReturnType dsl_ode_trigger_tracker_confidence_min_set(const wchar_t* name, 
    float min_confidence)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerTrackerConfidenceMinSet(
        cstrName.c_str(), min_confidence);
}

DslReturnType dsl_ode_trigger_tracker_confidence_max_get(const wchar_t* name, 
    float* max_confidence)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerTrackerConfidenceMaxGet(
        cstrName.c_str(), max_confidence);
}

DslReturnType dsl_ode_trigger_tracker_confidence_max_set(const wchar_t* name, 
    float max_confidence)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerTrackerConfidenceMaxSet(
        cstrName.c_str(), max_confidence);
}

DslReturnType dsl_ode_trigger_dimensions_min_get(const wchar_t* name, 
    float* min_width, float* min_height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDimensionsMinGet(
        cstrName.c_str(), min_width, min_height);
}

DslReturnType dsl_ode_trigger_dimensions_min_set(const wchar_t* name, 
    float min_width, float min_height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDimensionsMinSet(
        cstrName.c_str(), min_width, min_height);
}

DslReturnType dsl_ode_trigger_dimensions_max_get(const wchar_t* name, 
    float* max_width, float* max_height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDimensionsMaxGet(
        cstrName.c_str(), max_width, max_height);
}

DslReturnType dsl_ode_trigger_dimensions_max_set(const wchar_t* name, 
    float max_width, float max_height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDimensionsMaxSet(
        cstrName.c_str(), max_width, max_height);
}

DslReturnType dsl_ode_trigger_infer_done_only_get(const wchar_t* name, 
    boolean* infer_done_only)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerInferDoneOnlyGet(
        cstrName.c_str(), infer_done_only);
}

DslReturnType dsl_ode_trigger_infer_done_only_set(const wchar_t* name, 
    boolean infer_done_only)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerInferDoneOnlySet(
        cstrName.c_str(), infer_done_only);
}

DslReturnType dsl_ode_trigger_frame_count_min_get(const wchar_t* name, 
    uint* min_count_n, uint* min_count_d)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerFrameCountMinGet(cstrName.c_str(), 
        min_count_n, min_count_d);
}

DslReturnType dsl_ode_trigger_frame_count_min_set(const wchar_t* name, 
    uint min_count_n, uint min_count_d)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerFrameCountMinSet(cstrName.c_str(), 
        min_count_n, min_count_d);
}

DslReturnType dsl_ode_trigger_interval_get(const wchar_t* name, uint* interval)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerIntervalGet(cstrName.c_str(), interval);
}

DslReturnType dsl_ode_trigger_interval_set(const wchar_t* name, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerIntervalSet(cstrName.c_str(), interval);
}

DslReturnType dsl_ode_trigger_action_add(const wchar_t* name, const wchar_t* action)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(action);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAction(action);
    std::string cstrAction(wstrAction.begin(), wstrAction.end());

    return DSL::Services::GetServices()->OdeTriggerActionAdd(cstrName.c_str(), cstrAction.c_str());
}

DslReturnType dsl_ode_trigger_action_add_many(const wchar_t* name, const wchar_t** actions)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(actions);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    for (const wchar_t** action = actions; *action; action++)
    {
        std::wstring wstrAction(*action);
        std::string cstrAction(wstrAction.begin(), wstrAction.end());
        
        DslReturnType retval = DSL::Services::GetServices()->
            OdeTriggerActionAdd(cstrName.c_str(), cstrAction.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_ode_trigger_action_remove(const wchar_t* name, const wchar_t* action)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(action);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAction(action);
    std::string cstrAction(wstrAction.begin(), wstrAction.end());

    return DSL::Services::GetServices()->OdeTriggerActionRemove(cstrName.c_str(), cstrAction.c_str());
}

DslReturnType dsl_ode_trigger_action_remove_many(const wchar_t* name, const wchar_t** actions)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(actions);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    for (const wchar_t** action = actions; *action; action++)
    {
        std::wstring wstrAction(*action);
        std::string cstrAction(wstrAction.begin(), wstrAction.end());
        
        DslReturnType retval = DSL::Services::GetServices()->
            OdeTriggerActionRemove(cstrName.c_str(), cstrAction.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_ode_trigger_action_remove_all(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->OdeTriggerActionRemoveAll(cstrName.c_str());
}

DslReturnType dsl_ode_trigger_area_add(const wchar_t* name, const wchar_t* area)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(area);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrArea(area);
    std::string cstrArea(wstrArea.begin(), wstrArea.end());

    return DSL::Services::GetServices()->OdeTriggerAreaAdd(cstrName.c_str(), cstrArea.c_str());
}

DslReturnType dsl_ode_trigger_area_add_many(const wchar_t* name, const wchar_t** areas)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(areas);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    for (const wchar_t** area = areas; *area; area++)
    {
        std::wstring wstrArea(*area);
        std::string cstrArea(wstrArea.begin(), wstrArea.end());
        
        DslReturnType retval = DSL::Services::GetServices()->
            OdeTriggerAreaAdd(cstrName.c_str(), cstrArea.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_ode_trigger_area_remove(const wchar_t* name, const wchar_t* area)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(area);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrArea(area);
    std::string cstrArea(wstrArea.begin(), wstrArea.end());

    return DSL::Services::GetServices()->OdeTriggerAreaRemove(cstrName.c_str(), cstrArea.c_str());
}

DslReturnType dsl_ode_trigger_area_remove_many(const wchar_t* name, const wchar_t** areas)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(areas);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    for (const wchar_t** area = areas; *area; area++)
    {
        std::wstring wstrArea(*area);
        std::string cstrArea(wstrArea.begin(), wstrArea.end());
        
        DslReturnType retval = DSL::Services::GetServices()->
            OdeTriggerAreaRemove(cstrName.c_str(), cstrArea.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_ode_trigger_area_remove_all(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->OdeTriggerAreaRemoveAll(cstrName.c_str());
}

DslReturnType dsl_ode_trigger_accumulator_add(const wchar_t* name, 
    const wchar_t* accumulator)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(accumulator);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAccumulator(accumulator);
    std::string cstrAccumulator(wstrAccumulator.begin(), wstrAccumulator.end());

    return DSL::Services::GetServices()->OdeTriggerAccumulatorAdd(
        cstrName.c_str(), cstrAccumulator.c_str());
}

DslReturnType dsl_ode_trigger_accumulator_remove(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerAccumulatorRemove(
        cstrName.c_str());
}

DslReturnType dsl_ode_trigger_heat_mapper_add(const wchar_t* name, 
    const wchar_t* heat_mapper)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(heat_mapper);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHeatMapper(heat_mapper);
    std::string cstrHeatMapper(wstrHeatMapper.begin(), wstrHeatMapper.end());

    return DSL::Services::GetServices()->OdeTriggerHeatMapperAdd(
        cstrName.c_str(), cstrHeatMapper.c_str());
}

DslReturnType dsl_ode_trigger_heat_mapper_remove(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerHeatMapperRemove(
        cstrName.c_str());
}

DslReturnType dsl_ode_trigger_delete(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDelete(cstrName.c_str());
}

DslReturnType dsl_ode_trigger_delete_many(const wchar_t** names)
{
    RETURN_IF_PARAM_IS_NULL(names);

    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());
        DslReturnType retval = DSL::Services::GetServices()->OdeTriggerDelete(cstrName.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_ode_trigger_delete_all()
{
    return DSL::Services::GetServices()->OdeTriggerDeleteAll();
}

uint dsl_ode_trigger_list_size()
{
    return DSL::Services::GetServices()->OdeTriggerListSize();
}

DslReturnType dsl_ode_accumulator_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeAccumulatorNew(cstrName.c_str());
}

DslReturnType dsl_ode_accumulator_action_add(const wchar_t* name, const wchar_t* action)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(action);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAction(action);
    std::string cstrAction(wstrAction.begin(), wstrAction.end());

    return DSL::Services::GetServices()->OdeAccumulatorActionAdd(cstrName.c_str(), 
        cstrAction.c_str());
}

DslReturnType dsl_ode_accumulator_action_add_many(const wchar_t* name, const wchar_t** actions)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(actions);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    for (const wchar_t** action = actions; *action; action++)
    {
        std::wstring wstrAction(*action);
        std::string cstrAction(wstrAction.begin(), wstrAction.end());
        
        DslReturnType retval = DSL::Services::GetServices()->
            OdeAccumulatorActionAdd(cstrName.c_str(), cstrAction.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_ode_accumulator_action_remove(const wchar_t* name, const wchar_t* action)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(action);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAction(action);
    std::string cstrAction(wstrAction.begin(), wstrAction.end());

    return DSL::Services::GetServices()->OdeAccumulatorActionRemove(cstrName.c_str(), cstrAction.c_str());
}

DslReturnType dsl_ode_accumulator_action_remove_many(const wchar_t* name, const wchar_t** actions)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(actions);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    for (const wchar_t** action = actions; *action; action++)
    {
        std::wstring wstrAction(*action);
        std::string cstrAction(wstrAction.begin(), wstrAction.end());
        
        DslReturnType retval = DSL::Services::GetServices()->
            OdeAccumulatorActionRemove(cstrName.c_str(), cstrAction.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_ode_accumulator_action_remove_all(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->OdeAccumulatorActionRemoveAll(cstrName.c_str());
}

DslReturnType dsl_ode_accumulator_delete(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeAccumulatorDelete(cstrName.c_str());
}

DslReturnType dsl_ode_accumulator_delete_many(const wchar_t** names)
{
    RETURN_IF_PARAM_IS_NULL(names);

    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());
        DslReturnType retval = DSL::Services::GetServices()->OdeAccumulatorDelete(cstrName.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_ode_accumulator_delete_all()
{
    return DSL::Services::GetServices()->OdeAccumulatorDeleteAll();
}

uint dsl_ode_accumulator_list_size()
{
    return DSL::Services::GetServices()->OdeAccumulatorListSize();
}

DslReturnType dsl_ode_heat_mapper_new(const wchar_t* name, 
    uint cols, uint rows, uint bbox_test_point, const wchar_t* color_palette)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(color_palette);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColorPalette(color_palette);
    std::string cstrColorPalette(wstrColorPalette.begin(), wstrColorPalette.end());

    return DSL::Services::GetServices()->OdeHeatMapperNew(
        cstrName.c_str(), cols, rows, bbox_test_point, cstrColorPalette.c_str());
}

DslReturnType dsl_ode_heat_mapper_color_palette_get(const wchar_t* name, 
    const wchar_t** color_palette)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(color_palette);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cColorPalette;
    static std::string cstrColorPalette;
    static std::wstring wcstrColorPalette;
    
    DslReturnType result = DSL::Services::GetServices()->OdeHeatMapperColorPaletteGet(
        cstrName.c_str(), &cColorPalette);
    if (result == DSL_RESULT_SUCCESS)
    {
        cstrColorPalette.assign(cColorPalette);
        wcstrColorPalette.assign(cstrColorPalette.begin(), cstrColorPalette.end());
        *color_palette = wcstrColorPalette.c_str();
    }
    return result;
}

DslReturnType dsl_ode_heat_mapper_color_palette_set(const wchar_t* name, 
    const wchar_t* color_palette)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(color_palette);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColorPalette(color_palette);
    std::string cstrColorPalette(wstrColorPalette.begin(), wstrColorPalette.end());

    return DSL::Services::GetServices()->OdeHeatMapperColorPaletteSet(
        cstrName.c_str(), cstrColorPalette.c_str());
}
    
DslReturnType dsl_ode_heat_mapper_legend_settings_get(const wchar_t* name, 
    boolean* enabled, uint* location, uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeHeatMapperLegendSettingsGet(
        cstrName.c_str(), enabled, location, width, height);
}
    
DslReturnType dsl_ode_heat_mapper_legend_settings_set(const wchar_t* name, 
    boolean enabled, uint location, uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeHeatMapperLegendSettingsSet(
        cstrName.c_str(), enabled, location, width, height);
}

DslReturnType dsl_ode_heat_mapper_metrics_clear(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeHeatMapperMetricsClear(
        cstrName.c_str());
}

DslReturnType dsl_ode_heat_mapper_metrics_get(const wchar_t* name,
    const uint64_t** buffer, uint* size)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(buffer);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeHeatMapperMetricsGet(
        cstrName.c_str(), buffer, size);
}

DslReturnType dsl_ode_heat_mapper_metrics_print(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeHeatMapperMetricsPrint(
        cstrName.c_str());
}

DslReturnType dsl_ode_heat_mapper_metrics_log(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeHeatMapperMetricsLog(
        cstrName.c_str());
}

DslReturnType dsl_ode_heat_mapper_metrics_file(const wchar_t* name,
    const wchar_t* file_path, uint mode, uint format)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->OdeHeatMapperMetricsFile(
        cstrName.c_str(), cstrFilePath.c_str(), mode, format);
}
    
DslReturnType dsl_ode_heat_mapper_delete(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeHeatMapperDelete(cstrName.c_str());
}

DslReturnType dsl_ode_heat_mapper_delete_many(const wchar_t** names)
{
    RETURN_IF_PARAM_IS_NULL(names);

    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());
        DslReturnType retval = DSL::Services::GetServices()->
            OdeHeatMapperDelete(cstrName.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_ode_heat_mapper_delete_all()
{
    return DSL::Services::GetServices()->OdeHeatMapperDeleteAll();
}

uint dsl_ode_heat_mapper_list_size()
{
    return DSL::Services::GetServices()->OdeHeatMapperListSize();
}

DslReturnType dsl_pph_custom_new(const wchar_t* name,
     dsl_pph_custom_client_handler_cb client_handler, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(client_handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PphCustomNew(cstrName.c_str(), 
        client_handler, client_data);
}

DslReturnType dsl_pph_meter_new(const wchar_t* name, uint interval,
    dsl_pph_meter_client_handler_cb client_handler, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(client_handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PphMeterNew(cstrName.c_str(),
        interval, client_handler, client_data);
}

DslReturnType dsl_pph_meter_interval_get(const wchar_t* name, uint* interval)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PphMeterIntervalGet(cstrName.c_str(), interval);
}

DslReturnType dsl_pph_meter_interval_set(const wchar_t* name, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PphMeterIntervalSet(cstrName.c_str(), interval);
}

DslReturnType dsl_pph_ode_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PphOdeNew(cstrName.c_str());
}

DslReturnType dsl_pph_ode_trigger_add(const wchar_t* name, const wchar_t* trigger)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(trigger);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOdeTrigger(trigger);
    std::string cstrOdeTrigger(wstrOdeTrigger.begin(), wstrOdeTrigger.end());

    return DSL::Services::GetServices()->PphOdeTriggerAdd(cstrName.c_str(), cstrOdeTrigger.c_str());
}

DslReturnType dsl_pph_ode_trigger_add_many(const wchar_t* name, const wchar_t** triggers)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(triggers);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    for (const wchar_t** trigger = triggers; *trigger; trigger++)
    {
        std::wstring wstrOdeTrigger(*trigger);
        std::string cstrOdeTrigger(wstrOdeTrigger.begin(), wstrOdeTrigger.end());
        DslReturnType retval = DSL::Services::GetServices()->
            PphOdeTriggerAdd(cstrName.c_str(), cstrOdeTrigger.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_pph_ode_trigger_remove(const wchar_t* name, const wchar_t* trigger)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(trigger);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOdeTrigger(trigger);
    std::string cstrOdeTrigger(wstrOdeTrigger.begin(), wstrOdeTrigger.end());

    return DSL::Services::GetServices()->PphOdeTriggerRemove(cstrName.c_str(), cstrOdeTrigger.c_str());
}

DslReturnType dsl_pph_ode_trigger_remove_many(const wchar_t* name, const wchar_t** triggers)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(triggers);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    for (const wchar_t** trigger = triggers; *trigger; trigger++)
    {
        std::wstring wstrOdeTrigger(*trigger);
        std::string cstrOdeTrigger(wstrOdeTrigger.begin(), wstrOdeTrigger.end());
        DslReturnType retval = DSL::Services::GetServices()->PphOdeTriggerRemove(cstrName.c_str(), cstrOdeTrigger.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_pph_ode_trigger_remove_all(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PphOdeTriggerRemoveAll(cstrName.c_str());
}

DslReturnType dsl_pph_ode_display_meta_alloc_size_get(const wchar_t* name, uint* size)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PphOdeDisplayMetaAllocSizeGet(
        cstrName.c_str(), size);
}

DslReturnType dsl_pph_ode_display_meta_alloc_size_set(const wchar_t* name, uint size)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PphOdeDisplayMetaAllocSizeSet(
        cstrName.c_str(), size);
}

DslReturnType dsl_pph_nmp_new(const wchar_t* name, const wchar_t* label_file,
    uint process_method, uint match_method, float match_threshold)
{
#if !defined(BUILD_NMP_PPH)
    #error "BUILD_NMP_PPH must be defined"
#elif BUILD_NMP_PPH != true
    LOG_ERROR("To use the NMS PPH services, set BUILD_NMP_PPH=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrLabelFile;
    if (label_file != NULL)
    {
        std::wstring wstrLabelFile(label_file);
        cstrLabelFile.assign(wstrLabelFile.begin(), wstrLabelFile.end());
    }

    return DSL::Services::GetServices()->PphNmpNew(cstrName.c_str(),
        cstrLabelFile.c_str(), process_method, match_method, match_threshold);
#endif  
}

DslReturnType dsl_pph_nmp_label_file_get(const wchar_t* name, 
     const wchar_t** label_file)
{
#if !defined(BUILD_NMP_PPH)
    #error "BUILD_NMP_PPH must be defined"
#elif BUILD_NMP_PPH != true
    LOG_ERROR("To use the NMS PPH services, set BUILD_NMP_PPH=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(label_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cLabelFile;
    static std::string cstrLabelFile;
    static std::wstring wcstrLabelFile;
    
    uint retval = DSL::Services::GetServices()->PphNmpLabelFileGet(
        cstrName.c_str(), &cLabelFile);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrLabelFile.assign(cLabelFile);
        if (cstrLabelFile.size())
        {
            wcstrLabelFile.assign(cstrLabelFile.begin(), cstrLabelFile.end());
            *label_file = wcstrLabelFile.c_str();
        }
        else
        {
            *label_file = NULL;
        }
    }
    return retval;
#endif    
}
 
DslReturnType dsl_pph_nmp_label_file_set(const wchar_t* name, 
     const wchar_t* label_file)
{
#if !defined(BUILD_NMP_PPH)
    #error "BUILD_NMP_PPH must be defined"
#elif BUILD_NMP_PPH != true
    LOG_ERROR("To use the NMS PPH services, set BUILD_NMP_PPH=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrLabelFile;
    if (label_file != NULL)
    {
        std::wstring wstrLabelFile(label_file);
        cstrLabelFile.assign(wstrLabelFile.begin(), wstrLabelFile.end());
    }

    return DSL::Services::GetServices()->PphNmpLabelFileSet(
        cstrName.c_str(), cstrLabelFile.c_str());
#endif
}
     
DslReturnType dsl_pph_nmp_process_method_get(const wchar_t* name, 
     uint* process_method)
{
#if !defined(BUILD_NMP_PPH)
    #error "BUILD_NMP_PPH must be defined"
#elif BUILD_NMP_PPH != true
    LOG_ERROR("To use the NMS PPH services, set BUILD_NMP_PPH=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(process_method);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->PphNmpProcessMethodGet(
        cstrName.c_str(), process_method);
#endif    
}
     
DslReturnType dsl_pph_nmp_process_method_set(const wchar_t* name, 
     uint process_method)
{
#if !defined(BUILD_NMP_PPH)
    #error "BUILD_NMP_PPH must be defined"
#elif BUILD_NMP_PPH != true
    LOG_ERROR("To use the NMS PPH services, set BUILD_NMP_PPH=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->PphNmpProcessMethodSet(
        cstrName.c_str(), process_method);
#endif    
}
     
DslReturnType dsl_pph_nmp_match_settings_get(const wchar_t* name,
    uint* match_method, float* match_threshold)
{
#if !defined(BUILD_NMP_PPH)
    #error "BUILD_NMP_PPH must be defined"
#elif BUILD_NMP_PPH != true
    LOG_ERROR("To use the NMS PPH services, set BUILD_NMP_PPH=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(match_method);
    RETURN_IF_PARAM_IS_NULL(match_threshold);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->PphNmpMatchSettingsGet(
        cstrName.c_str(), match_method, match_threshold);
#endif    
}
     
DslReturnType dsl_pph_nmp_match_settings_set(const wchar_t* name,
    uint match_method, float match_threshold)
{
#if !defined(BUILD_NMP_PPH)
    #error "BUILD_NMP_PPH must be defined"
#elif BUILD_NMP_PPH != true
    LOG_ERROR("To use the NMS PPH services, set BUILD_NMP_PPH=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->PphNmpMatchSettingsSet(
        cstrName.c_str(), match_method, match_threshold);
#endif    
}

DslReturnType dsl_pph_buffer_timeout_new(const wchar_t* name,
    uint timeout, dsl_pph_buffer_timeout_handler_cb handler, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PphBufferTimeoutNew(cstrName.c_str(), 
        timeout, handler, client_data);
}

DslReturnType dsl_pph_eos_new(const wchar_t* name,
    dsl_pph_eos_handler_cb handler, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PphEosNew(cstrName.c_str(), 
        handler, client_data);
}

DslReturnType dsl_pph_stream_event_new(const wchar_t* name,
    dsl_pph_stream_event_handler_cb handler, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PphStreamEventNew(cstrName.c_str(), 
        handler, client_data);
}
     
DslReturnType dsl_pph_enabled_get(const wchar_t* name, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PphEnabledGet(cstrName.c_str(), enabled);
}

DslReturnType dsl_pph_enabled_set(const wchar_t* name, boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PphEnabledSet(cstrName.c_str(), enabled);
}

DslReturnType dsl_pph_delete(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PphDelete(cstrName.c_str());
}

DslReturnType dsl_pph_delete_many(const wchar_t** names)
{
    RETURN_IF_PARAM_IS_NULL(names);

    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());
        DslReturnType retval = DSL::Services::GetServices()->PphDelete(cstrName.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_pph_delete_all()
{
    return DSL::Services::GetServices()->PphDeleteAll();
}

uint dsl_pph_list_size()
{
    return DSL::Services::GetServices()->PphListSize();
}

DslReturnType dsl_gst_element_new(const wchar_t* name, const wchar_t* factory_name)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(factory_name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFactoryName(factory_name);
    std::string cstrFactoryName(wstrFactoryName.begin(), 
        wstrFactoryName.end());

    return DSL::Services::GetServices()->GstElementNew(cstrName.c_str(), 
        cstrFactoryName.c_str());
}

DslReturnType dsl_gst_element_delete(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->GstElementDelete(cstrName.c_str());
}

DslReturnType dsl_gst_element_delete_many(const wchar_t** names)
{
    RETURN_IF_PARAM_IS_NULL(names);

    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());

        DslReturnType retval = DSL::Services::GetServices()->GstElementDelete(cstrName.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_gst_element_delete_all()
{
    return DSL::Services::GetServices()->GstElementDeleteAll();
}

uint dsl_gst_element_list_size()
{
    return DSL::Services::GetServices()->GstElementListSize();
}

DslReturnType dsl_gst_element_get(const wchar_t* name, void** element)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(element);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->GstElementGet(cstrName.c_str(), 
        element);
}

    DslReturnType dsl_gst_element_property_boolean_get(const wchar_t* name, 
    const wchar_t* property, boolean* value)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(property);
    RETURN_IF_PARAM_IS_NULL(value);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrProperty(property);
    std::string cstrProperty(wstrProperty.begin(), wstrProperty.end());

    return DSL::Services::GetServices()->GstElementPropertyBooleanGet(
        cstrName.c_str(), cstrProperty.c_str(), value);
}
    
DslReturnType dsl_gst_element_property_boolean_set(const wchar_t* name, 
    const wchar_t* property, boolean value)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(property);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrProperty(property);
    std::string cstrProperty(wstrProperty.begin(), wstrProperty.end());

    return DSL::Services::GetServices()->GstElementPropertyBooleanSet(
        cstrName.c_str(), cstrProperty.c_str(), value);
}
    
DslReturnType dsl_gst_element_property_float_get(const wchar_t* name, 
    const wchar_t* property, float* value)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(property);
    RETURN_IF_PARAM_IS_NULL(value);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrProperty(property);
    std::string cstrProperty(wstrProperty.begin(), wstrProperty.end());

    return DSL::Services::GetServices()->GstElementPropertyFloatGet(
        cstrName.c_str(), cstrProperty.c_str(), value);
}
    
DslReturnType dsl_gst_element_property_float_set(const wchar_t* name, 
    const wchar_t* property, float value)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(property);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrProperty(property);
    std::string cstrProperty(wstrProperty.begin(), wstrProperty.end());

    return DSL::Services::GetServices()->GstElementPropertyFloatSet(
        cstrName.c_str(), cstrProperty.c_str(), value);
}

DslReturnType dsl_gst_element_property_uint_get(const wchar_t* name, 
    const wchar_t* property, uint* value)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(property);
    RETURN_IF_PARAM_IS_NULL(value);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrProperty(property);
    std::string cstrProperty(wstrProperty.begin(), wstrProperty.end());

    return DSL::Services::GetServices()->GstElementPropertyUintGet(
        cstrName.c_str(), cstrProperty.c_str(), value);
}
    
DslReturnType dsl_gst_element_property_uint_set(const wchar_t* name, 
    const wchar_t* property, uint value)    
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(property);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrProperty(property);
    std::string cstrProperty(wstrProperty.begin(), wstrProperty.end());

    return DSL::Services::GetServices()->GstElementPropertyUintSet(
        cstrName.c_str(), cstrProperty.c_str(), value);
}
    
DslReturnType dsl_gst_element_property_int_get(const wchar_t* name, 
    const wchar_t* property, int* value)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(property);
    RETURN_IF_PARAM_IS_NULL(value);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrProperty(property);
    std::string cstrProperty(wstrProperty.begin(), wstrProperty.end());

    return DSL::Services::GetServices()->GstElementPropertyIntGet(
        cstrName.c_str(), cstrProperty.c_str(), value);
}
    
DslReturnType dsl_gst_element_property_int_set(const wchar_t* name, 
    const wchar_t* property, int value)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(property);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrProperty(property);
    std::string cstrProperty(wstrProperty.begin(), wstrProperty.end());

    return DSL::Services::GetServices()->GstElementPropertyIntSet(
        cstrName.c_str(), cstrProperty.c_str(), value);
}
    

DslReturnType dsl_gst_element_property_uint64_get(const wchar_t* name, 
    const wchar_t* property, uint64_t* value)    
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(property);
    RETURN_IF_PARAM_IS_NULL(value);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrProperty(property);
    std::string cstrProperty(wstrProperty.begin(), wstrProperty.end());

    return DSL::Services::GetServices()->GstElementPropertyUint64Get(
        cstrName.c_str(), cstrProperty.c_str(), value);
}
    
DslReturnType dsl_gst_element_property_uint64_set(const wchar_t* name, 
    const wchar_t* property, uint64_t value)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(property);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrProperty(property);
    std::string cstrProperty(wstrProperty.begin(), wstrProperty.end());

    return DSL::Services::GetServices()->GstElementPropertyUint64Set(
        cstrName.c_str(), cstrProperty.c_str(), value);
}
    
DslReturnType dsl_gst_element_property_int64_get(const wchar_t* name, 
    const wchar_t* property, int64_t* value)    
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(property);
    RETURN_IF_PARAM_IS_NULL(value);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrProperty(property);
    std::string cstrProperty(wstrProperty.begin(), wstrProperty.end());

    return DSL::Services::GetServices()->GstElementPropertyInt64Get(
        cstrName.c_str(), cstrProperty.c_str(), value);
}
    
DslReturnType dsl_gst_element_property_int64_set(const wchar_t* name, 
    const wchar_t* property, int64_t value)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(property);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrProperty(property);
    std::string cstrProperty(wstrProperty.begin(), wstrProperty.end());

    return DSL::Services::GetServices()->GstElementPropertyInt64Set(
        cstrName.c_str(), cstrProperty.c_str(), value);
}

DslReturnType dsl_gst_element_property_string_get(const wchar_t* name, 
    const wchar_t* property, const wchar_t** value)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(property);
    RETURN_IF_PARAM_IS_NULL(value);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrProperty(property);
    std::string cstrProperty(wstrProperty.begin(), wstrProperty.end());

    const char* cValue;
    static std::string cstrValue;
    static std::wstring wcstrValue;
    
    uint retval = DSL::Services::GetServices()->GstElementPropertyStringGet(
        cstrName.c_str(), cstrProperty.c_str(), &cValue);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        if(*cValue)
        { 
            cstrValue.assign(cValue);
            wcstrValue.assign(cstrValue.begin(), cstrValue.end());
            *value = wcstrValue.c_str();
        }
        else
        {
            *value = NULL;
        }
    }
    return retval;
}
    
DslReturnType dsl_gst_element_property_string_set(const wchar_t* name, 
    const wchar_t* property, const wchar_t* value)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(property);
    RETURN_IF_PARAM_IS_NULL(value);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrProperty(property);
    std::string cstrProperty(wstrProperty.begin(), wstrProperty.end());
    std::wstring wstrValue(value);
    std::string cstrValue(wstrValue.begin(), wstrValue.end());

    return DSL::Services::GetServices()->GstElementPropertyStringSet(
        cstrName.c_str(), cstrProperty.c_str(), cstrValue.c_str());
}
    
DslReturnType dsl_gst_element_pph_add(const wchar_t* name, 
    const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->GstElementPphAdd(cstrName.c_str(), 
        cstrHandler.c_str(), pad);
}

DslReturnType dsl_gst_element_pph_remove(const wchar_t* name,
    const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->GstElementPphRemove(cstrName.c_str(), 
        cstrHandler.c_str(), pad);
}

DslReturnType dsl_gst_bin_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->GstBinNew(cstrName.c_str());
}

DslReturnType dsl_gst_bin_new_element_add_many(const wchar_t* name, 
    const wchar_t** elements)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(elements);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    DslReturnType retval = DSL::Services::GetServices()->GstBinNew(
        cstrName.c_str());
    if (retval != DSL_RESULT_SUCCESS)
    {
        return retval;
    }
    for (const wchar_t** element = elements; *element; element++)
    {
        std::wstring wstrElement(*element);
        std::string cstrElement(wstrElement.begin(), wstrElement.end());
        DslReturnType retval = DSL::Services::GetServices()->GstBinElementAdd(
            cstrName.c_str(), cstrElement.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_gst_bin_element_add(const wchar_t* name, 
    const wchar_t* element)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(element);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrElement(element);
    std::string cstrElement(wstrElement.begin(), wstrElement.end());

    return DSL::Services::GetServices()->GstBinElementAdd(
        cstrName.c_str(), cstrElement.c_str());
}

DslReturnType dsl_gst_bin_element_add_many(const wchar_t* name, 
    const wchar_t** elements)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(elements);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    for (const wchar_t** element = elements; *element; element++)
    {
        std::wstring wstrElement(*element);
        std::string cstrElement(wstrElement.begin(), wstrElement.end());
        DslReturnType retval = DSL::Services::GetServices()->GstBinElementAdd(
            cstrName.c_str(), cstrElement.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_gst_bin_element_remove(const wchar_t* name, 
    const wchar_t* element)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(element);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrElement(element);
    std::string cstrElement(wstrElement.begin(), wstrElement.end());

    return DSL::Services::GetServices()->GstBinElementRemove(cstrName.c_str(), 
        cstrElement.c_str());
}

DslReturnType dsl_gst_bin_element_remove_many(const wchar_t* name, 
    const wchar_t** elements)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(elements);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    for (const wchar_t** element = elements; *element; element++)
    {
        std::wstring wstrElement(*element);
        std::string cstrElement(wstrElement.begin(), wstrElement.end());
        DslReturnType retval = 
            DSL::Services::GetServices()->GstBinElementRemove(cstrName.c_str(), 
                cstrElement.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_source_app_new(const wchar_t* name, boolean is_live, 
    const wchar_t* buffer_in_format, uint width, uint height, uint fps_n, uint fps_d)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(buffer_in_format);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrBufferInFormat(buffer_in_format);
    std::string cstrBufferInFormat(wstrBufferInFormat.begin(), 
        wstrBufferInFormat.end());

    return DSL::Services::GetServices()->SourceAppNew(cstrName.c_str(), 
        is_live, cstrBufferInFormat.c_str(), width, height, fps_n, fps_d);
}

DslReturnType dsl_source_custom_new(const wchar_t* name, const wchar_t* elementName,
                                    const wchar_t* factoryName, void** element) 
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(elementName);
    RETURN_IF_PARAM_IS_NULL(factoryName);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrElement(elementName);
    std::string cstrElement(wstrElement.begin(),wstrElement.end());
    std::wstring wstrFactory(factoryName);
    std::string cstrFactory(wstrFactory.begin(), wstrFactory.end());

    return DSL::Services::GetServices()->SourceCustomNew(cstrName.c_str(), cstrElement.c_str(), 
        cstrFactory.c_str(), element);                                        
}

DslReturnType dsl_source_app_data_handlers_add(const wchar_t* name, 
    dsl_source_app_need_data_handler_cb need_data_handler, 
    dsl_source_app_enough_data_handler_cb enough_data_handler, 
    void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(need_data_handler);
    RETURN_IF_PARAM_IS_NULL(enough_data_handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceAppDataHandlersAdd(cstrName.c_str(), 
        need_data_handler, enough_data_handler, client_data);
}

DslReturnType dsl_source_app_data_handlers_remove(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceAppDataHandlersRemove(cstrName.c_str());
}
 
DslReturnType dsl_source_app_buffer_push(const wchar_t* name, void* buffer)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(buffer);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceAppBufferPush(cstrName.c_str(), 
        buffer);
}

DslReturnType dsl_source_app_sample_push(const wchar_t* name, void* sample)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(sample);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceAppSamplePush(cstrName.c_str(), 
        sample);
}

DslReturnType dsl_source_app_eos(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceAppEos(cstrName.c_str());
}

DslReturnType dsl_source_app_stream_format_get(const wchar_t* name, 
    uint* stream_format)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(stream_format);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceAppStreamFormatGet(
        cstrName.c_str(), stream_format);
}
    
DslReturnType dsl_source_app_stream_format_set(const wchar_t* name, 
    uint stream_format)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceAppStreamFormatSet(
        cstrName.c_str(), stream_format);
}
    
DslReturnType dsl_source_app_block_enabled_get(const wchar_t* name, 
    boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceAppBlockEnabledGet(
        cstrName.c_str(), enabled);
}
   
DslReturnType dsl_source_app_block_enabled_set(const wchar_t* name, 
    boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceAppBlockEnabledSet(
        cstrName.c_str(), enabled);
}

DslReturnType dsl_source_app_current_level_bytes_get(const wchar_t* name,
    uint64_t* level)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(level);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceAppCurrentLevelBytesGet(
        cstrName.c_str(), level);
}
   
DslReturnType dsl_source_app_max_level_bytes_get(const wchar_t* name,
    uint64_t* level)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(level);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceAppMaxLevelBytesGet(
        cstrName.c_str(), level);
}
   
DslReturnType dsl_source_app_max_level_bytes_set(const wchar_t* name,
    uint64_t level)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceAppMaxLevelBytesSet(
        cstrName.c_str(), level);
}
 
//DslReturnType dsl_source_app_leaky_type_get(const wchar_t* name,
//    uint* leaky_type)
//{
//    RETURN_IF_PARAM_IS_NULL(name);
//    RETURN_IF_PARAM_IS_NULL(leaky_type);
//
//    std::wstring wstrName(name);
//    std::string cstrName(wstrName.begin(), wstrName.end());
//
//    return DSL::Services::GetServices()->SourceAppLeakyTypeGet(cstrName.c_str(),
//        leaky_type);
//}
//
//DslReturnType dsl_source_app_leaky_type_set(const wchar_t* name,
//    uint leaky_type)
//{
//    RETURN_IF_PARAM_IS_NULL(name);
//
//    std::wstring wstrName(name);
//    std::string cstrName(wstrName.begin(), wstrName.end());
//
//    return DSL::Services::GetServices()->SourceAppLeakyTypeSet(cstrName.c_str(),
//        leaky_type);
//}
  
DslReturnType dsl_source_csi_new(const wchar_t* name, 
    uint width, uint height, uint fps_n, uint fps_d)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceCsiNew(cstrName.c_str(), 
        width, height, fps_n, fps_d);
}

DslReturnType dsl_source_csi_sensor_id_get(const wchar_t* name,
    uint* sensor_id)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(sensor_id);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceCsiSensorIdGet(cstrName.c_str(), 
        sensor_id);
}

DslReturnType dsl_source_csi_sensor_id_set(const wchar_t* name,
    uint sensor_id)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceCsiSensorIdSet(cstrName.c_str(), 
        sensor_id);
}
    
DslReturnType dsl_source_v4l2_new(const wchar_t* name,
    const wchar_t* device_location)
{
    RETURN_IF_PARAM_IS_NULL(name);

    RETURN_IF_PARAM_IS_NULL(device_location);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrDeviceLocation(device_location);
    std::string cstrDeviceLocation(wstrDeviceLocation.begin(), 
        wstrDeviceLocation.end());

    return DSL::Services::GetServices()->SourceV4l2New(cstrName.c_str(), 
        cstrDeviceLocation.c_str());
}

DslReturnType dsl_source_v4l2_device_location_get(const wchar_t* name,
    const wchar_t** device_location)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(device_location);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cDeviceLocation;
    static std::string cstrDeviceLocation;
    static std::wstring wcstrDeviceLocation;
    
    uint retval = DSL::Services::GetServices()->
        SourceV4l2DeviceLocationGet(cstrName.c_str(), 
        &cDeviceLocation);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrDeviceLocation.assign(cDeviceLocation);
        wcstrDeviceLocation.assign(cstrDeviceLocation.begin(), 
            cstrDeviceLocation.end());
        *device_location = wcstrDeviceLocation.c_str();
    }
    return retval;
}
    

DslReturnType dsl_source_v4l2_device_location_set(const wchar_t* name,
    const wchar_t* device_location)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(device_location);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrDeviceLocation(device_location);
    std::string cstrDeviceLocation(wstrDeviceLocation.begin(), 
        wstrDeviceLocation.end());
    
    return DSL::Services::GetServices()->SourceV4l2DeviceLocationSet(
        cstrName.c_str(), cstrDeviceLocation.c_str());
}

DslReturnType dsl_source_v4l2_dimensions_set(const wchar_t* name, 
    uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceV4l2DimensionsSet(
        cstrName.c_str(), width, height);
}

DslReturnType dsl_source_v4l2_frame_rate_set(const wchar_t* name, 
    uint fps_n, uint fps_d)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceV4l2FrameRateSet(
        cstrName.c_str(), fps_n, fps_d);
}
    
DslReturnType dsl_source_v4l2_device_name_get(const wchar_t* name,
    const wchar_t** device_name)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(device_name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cDeviceName;
    static std::string cstrDeviceName;
    static std::wstring wcstrDeviceName;
    
    uint retval = DSL::Services::GetServices()->SourceV4l2DeviceNameGet(
        cstrName.c_str(), &cDeviceName);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrDeviceName.assign(cDeviceName);
        wcstrDeviceName.assign(cstrDeviceName.begin(), cstrDeviceName.end());
        *device_name = wcstrDeviceName.c_str();
    }
    return retval;
}
    
DslReturnType dsl_source_v4l2_device_fd_get(const wchar_t* name,
    int* device_fd)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(device_fd);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SourceV4l2DeviceFdGet(
        cstrName.c_str(), device_fd);
}

DslReturnType dsl_source_v4l2_device_flags_get(const wchar_t* name,
    uint* device_flags)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(device_flags);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SourceV4l2DeviceFlagsGet(
        cstrName.c_str(), device_flags);
}

DslReturnType dsl_source_v4l2_picture_settings_get(const wchar_t* name,
    int* brightness, int* contrast, int* hue)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(brightness);
    RETURN_IF_PARAM_IS_NULL(contrast);
    RETURN_IF_PARAM_IS_NULL(hue);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SourceV4l2PictureSettingsGet(
        cstrName.c_str(), brightness, contrast, hue);
}

DslReturnType dsl_source_v4l2_picture_settings_set(const wchar_t* name,
    int brightness, int contrast, int hue)    
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SourceV4l2PictureSettingsSet(
        cstrName.c_str(), brightness, contrast, hue);
}

DslReturnType dsl_source_uri_new(const wchar_t* name, const wchar_t* uri, 
    boolean is_live, uint skip_frames, uint dropFrameInterval)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(uri);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUri(uri);
    std::string cstrUri(wstrUri.begin(), wstrUri.end());

    return DSL::Services::GetServices()->SourceUriNew(cstrName.c_str(), cstrUri.c_str(), 
        is_live, skip_frames, dropFrameInterval);
}

DslReturnType dsl_source_file_new(const wchar_t* name, 
    const wchar_t* file_path, boolean repeat_enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    std::string cstrFilePath;
    if (file_path != NULL)
    {
        std::wstring wstrFilePath(file_path);
        cstrFilePath.assign(wstrFilePath.begin(), wstrFilePath.end());
    }

    return DSL::Services::GetServices()->SourceFileNew(cstrName.c_str(), 
        cstrFilePath.c_str(), repeat_enabled);
}

DslReturnType dsl_source_file_file_path_get(const wchar_t* name, 
    const wchar_t** file_path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cFilePath;
    static std::string cstrFilePath;
    static std::wstring wcstrFilePath;
    
    uint retval = DSL::Services::GetServices()->SourceFileFilePathGet(
        cstrName.c_str(), &cFilePath);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrFilePath.assign(cFilePath);
        wcstrFilePath.assign(cstrFilePath.begin(), cstrFilePath.end());
        *file_path = wcstrFilePath.c_str();
    }
    return retval;
    
}

DslReturnType dsl_source_file_file_path_set(const wchar_t* name, 
    const wchar_t* file_path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->SourceFileFilePathSet(cstrName.c_str(), 
        cstrFilePath.c_str());
}

DslReturnType dsl_source_file_repeat_enabled_get(const wchar_t* name, 
    boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceFileRepeatEnabledGet(
        cstrName.c_str(), enabled);
}

DslReturnType dsl_source_file_repeat_enabled_set(const wchar_t* name, 
    boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceFileRepeatEnabledSet(cstrName.c_str(),
        enabled);
}

DslReturnType dsl_source_image_single_new(const wchar_t* name, 
    const wchar_t* file_path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->SourceImageNew(cstrName.c_str(), 
        cstrFilePath.c_str());
}

DslReturnType dsl_source_image_multi_new(const wchar_t* name, 
    const wchar_t* file_path, uint fps_n, uint fps_d)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->SourceImageMultiNew(cstrName.c_str(), 
        cstrFilePath.c_str(), fps_n, fps_d);
}

DslReturnType dsl_source_image_multi_loop_enabled_get(const wchar_t* name, 
    boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceImageMultiLoopEnabledGet(
        cstrName.c_str(), enabled);
}
    
DslReturnType dsl_source_image_multi_loop_enabled_set(const wchar_t* name, 
    boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceImageMultiLoopEnabledSet(
        cstrName.c_str(), enabled);
}

DslReturnType dsl_source_image_multi_indices_get(const wchar_t* name, 
    int* start_index, int* stop_index)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(start_index);
    RETURN_IF_PARAM_IS_NULL(stop_index);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceImageMultiIndicesGet(
        cstrName.c_str(), start_index, stop_index);
}

DslReturnType dsl_source_image_multi_indices_set(const wchar_t* name, 
    int start_index, int stop_index)
{
    RETURN_IF_PARAM_IS_NULL(name);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceImageMultiIndicesSet(
        cstrName.c_str(), start_index, stop_index);
}
    
DslReturnType dsl_source_image_stream_new(const wchar_t* name, 
    const wchar_t* file_path, boolean is_live, uint fps_n, uint fps_d, uint timeout)
{
#if !defined(BUILD_WITH_FFMPEG) || !defined(BUILD_WITH_OPENCV)
    #error "BUILD_WITH_FFMPEG and BUILD_WITH_OPENCV must be defined"
#elif (BUILD_WITH_FFMPEG != true) && (BUILD_WITH_OPENCV != true)
    LOG_ERROR("dsl_sink_frame_capture_new requires one of BUILD_WITH_FFMPEG \
       or BUILD_WITH_OPENCV to be set true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else    
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->SourceImageStreamNew(cstrName.c_str(), 
        cstrFilePath.c_str(), is_live, fps_n, fps_d, timeout);
#endif        
}

DslReturnType dsl_source_image_stream_timeout_get(const wchar_t* name, uint* timeout)
{
#if !defined(BUILD_WITH_FFMPEG) || !defined(BUILD_WITH_OPENCV)
    #error "BUILD_WITH_FFMPEG and BUILD_WITH_OPENCV must be defined"
#elif (BUILD_WITH_FFMPEG != true) && (BUILD_WITH_OPENCV != true)
    LOG_ERROR("dsl_sink_frame_capture_new requires one of BUILD_WITH_FFMPEG \
       or BUILD_WITH_OPENCV to be set true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else    
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceImageStreamTimeoutGet(cstrName.c_str(),
        timeout);
#endif        
}

DslReturnType dsl_source_image_stream_timeout_set(const wchar_t* name, uint timeout)
{
#if !defined(BUILD_WITH_FFMPEG) || !defined(BUILD_WITH_OPENCV)
    #error "BUILD_WITH_FFMPEG and BUILD_WITH_OPENCV must be defined"
#elif (BUILD_WITH_FFMPEG != true) && (BUILD_WITH_OPENCV != true)
    LOG_ERROR("dsl_sink_frame_capture_new requires one of BUILD_WITH_FFMPEG \
       or BUILD_WITH_OPENCV to be set true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else    
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceImageStreamTimeoutSet(cstrName.c_str(),
        timeout);
#endif
}

DslReturnType dsl_source_image_file_path_get(const wchar_t* name, 
    const wchar_t** file_path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cFilePath;
    static std::string cstrFilePath;
    static std::wstring wcstrFilePath;
    
    uint retval = DSL::Services::GetServices()->SourceImageFilePathGet(
        cstrName.c_str(), &cFilePath);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrFilePath.assign(cFilePath);
        wcstrFilePath.assign(cstrFilePath.begin(), cstrFilePath.end());
        *file_path = wcstrFilePath.c_str();
    }
    return retval;
    
}

DslReturnType dsl_source_image_file_path_set(const wchar_t* name, 
    const wchar_t* file_path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->SourceImageFilePathSet(cstrName.c_str(), 
        cstrFilePath.c_str());
}

DslReturnType dsl_source_interpipe_new(const wchar_t* name, 
    const wchar_t* listen_to, boolean is_live,
    boolean accept_eos, boolean accept_events)
{
#if !defined(BUILD_INTER_PIPE)
    #error "BUILD_INTER_PIPE must be defined"
#elif BUILD_INTER_PIPE != true
    LOG_ERROR("To use the Inter-Pipe services, set BUILD_INTER_PIPE=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listen_to);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrListenTo(listen_to);
    std::string cstrListenTo(wstrListenTo.begin(), wstrListenTo.end());

    return DSL::Services::GetServices()->SourceInterpipeNew(cstrName.c_str(), 
        cstrListenTo.c_str(), is_live, accept_eos, accept_events);
#endif
}

DslReturnType dsl_source_interpipe_listen_to_get(const wchar_t* name, 
    const wchar_t** listen_to)
{
#if !defined(BUILD_INTER_PIPE)
    #error "BUILD_INTER_PIPE must be defined"
#elif BUILD_INTER_PIPE != true
    LOG_ERROR("To use the Inter-Pipe services, set BUILD_INTER_PIPE=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listen_to);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cListenTo;
    static std::string cstrListenTo;
    static std::wstring wcstrListenTo;
    
    uint retval = DSL::Services::GetServices()->SourceInterpipeListenToGet(cstrName.c_str(), 
        &cListenTo);

    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrListenTo.assign(cListenTo);
        wcstrListenTo.assign(cstrListenTo.begin(), cstrListenTo.end());
        *listen_to = wcstrListenTo.c_str();
    }
    return retval;
#endif
}

DslReturnType dsl_source_interpipe_listen_to_set(const wchar_t* name, 
    const wchar_t* listen_to)
{
#if !defined(BUILD_INTER_PIPE)
    #error "BUILD_INTER_PIPE must be defined"
#elif BUILD_INTER_PIPE != true
    LOG_ERROR("To use the Inter-Pipe services, set BUILD_INTER_PIPE=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listen_to);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrListenTo(listen_to);
    std::string cstrListenTo(wstrListenTo.begin(), wstrListenTo.end());

    return DSL::Services::GetServices()->SourceInterpipeListenToSet(cstrName.c_str(), 
        cstrListenTo.c_str());
#endif        
}    

DslReturnType dsl_source_interpipe_accept_settings_get(const wchar_t* name,
    boolean* accept_eos, boolean* accept_events)
{
#if !defined(BUILD_INTER_PIPE)
    #error "BUILD_INTER_PIPE must be defined"
#elif BUILD_INTER_PIPE != true
    LOG_ERROR("To use the Inter-Pipe services, set BUILD_INTER_PIPE=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(accept_eos);
    RETURN_IF_PARAM_IS_NULL(accept_events);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceInterpipeAcceptSettingsGet(
        cstrName.c_str(), accept_eos, accept_events);        
#endif        
}

DslReturnType dsl_source_interpipe_accept_settings_set(const wchar_t* name,
    boolean accept_eos, boolean accept_events)    
{
#if !defined(BUILD_INTER_PIPE)
    #error "BUILD_INTER_PIPE must be defined"
#elif BUILD_INTER_PIPE != true
    LOG_ERROR("To use the Inter-Pipe services, set BUILD_INTER_PIPE=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceInterpipeAcceptSettingsSet(
        cstrName.c_str(), accept_eos, accept_events);      
#endif
}
    
DslReturnType dsl_source_rtsp_new(const wchar_t* name, const wchar_t* uri, uint protocol, 
    uint skip_frames, uint dropFrameInterval, uint latency, uint timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(uri);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUri(uri);
    std::string cstrUri(wstrUri.begin(), wstrUri.end());

    return DSL::Services::GetServices()->SourceRtspNew(cstrName.c_str(), cstrUri.c_str(), 
        protocol, skip_frames, dropFrameInterval, latency, timeout);
}

DslReturnType dsl_source_duplicate_new(const wchar_t* name, const wchar_t* original)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(original);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOriginal(original);
    std::string cstrOriginal(wstrOriginal.begin(), wstrOriginal.end());
    
    return DSL::Services::GetServices()->SourceDuplicateNew(cstrName.c_str(), 
        cstrOriginal.c_str());
}

DslReturnType dsl_source_duplicate_original_get(const wchar_t* name, 
    const wchar_t** original)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(original);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cOriginal;
    static std::string cstrOriginal;
    static std::wstring wcstrOriginal;
    
    uint retval = DSL::Services::GetServices()->SourceDuplicateOriginalGet(
        cstrName.c_str(), &cOriginal);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrOriginal.assign(cOriginal);
        wcstrOriginal.assign(cstrOriginal.begin(), cstrOriginal.end());
        *original = wcstrOriginal.c_str();
    }
    return retval;
}
    
DslReturnType dsl_source_duplicate_original_set(const wchar_t* name, 
    const wchar_t* original)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(original);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOriginal(original);
    std::string cstrOriginal(wstrOriginal.begin(), wstrOriginal.end());
    
    return DSL::Services::GetServices()->SourceDuplicateOriginalSet(cstrName.c_str(), 
        cstrOriginal.c_str());
}
    
DslReturnType dsl_source_pph_add(const wchar_t* name, const wchar_t* handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->SourcePphAdd(cstrName.c_str(), 
        cstrHandler.c_str());
}

DslReturnType dsl_source_pph_remove(const wchar_t* name, const wchar_t* handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->SourcePphRemove(cstrName.c_str(), 
        cstrHandler.c_str());
}

DslReturnType dsl_source_media_type_get(const wchar_t* name,
    const wchar_t** media_type)   
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(media_type);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cMediaType;
    static std::string cstrMediaType;
    static std::wstring wcstrMediaType;
    
    uint retval = DSL::Services::GetServices()->SourceMediaTypeGet(
        cstrName.c_str(), &cMediaType);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrMediaType.assign(cMediaType);
        wcstrMediaType.assign(cstrMediaType.begin(), cstrMediaType.end());
        *media_type = wcstrMediaType.c_str();
    }
    return retval;
}
    
DslReturnType dsl_source_video_buffer_out_format_get(const wchar_t* name,
    const wchar_t** format)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(format);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cFormat;
    static std::string cstrFormat;
    static std::wstring wcstrFormat;
    
    uint retval = DSL::Services::GetServices()->SourceVideoBufferOutFormatGet(
        cstrName.c_str(), &cFormat);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrFormat.assign(cFormat);
        wcstrFormat.assign(cstrFormat.begin(), cstrFormat.end());
        *format = wcstrFormat.c_str();
    }
    return retval;
}
    
DslReturnType dsl_source_video_buffer_out_format_set(const wchar_t* name,
    const wchar_t* format)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(format);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::wstring wstrFormat(format);
    std::string cstrFormat(wstrFormat.begin(), wstrFormat.end());

    return DSL::Services::GetServices()->SourceVideoBufferOutFormatSet(
        cstrName.c_str(), cstrFormat.c_str());
}

DslReturnType dsl_source_video_buffer_out_dimensions_get(const wchar_t* name, 
    uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(width);
    RETURN_IF_PARAM_IS_NULL(height);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceVideoBufferOutDimensionsGet(
        cstrName.c_str(), width, height);
}

DslReturnType dsl_source_video_buffer_out_dimensions_set(const wchar_t* name, 
    uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceVideoBufferOutDimensionsSet(
        cstrName.c_str(), width, height);
}

DslReturnType dsl_source_video_buffer_out_frame_rate_get(const wchar_t* name, 
    uint* fps_n, uint* fps_d)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(fps_n);
    RETURN_IF_PARAM_IS_NULL(fps_d);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceVideoBufferOutFrameRateGet(
        cstrName.c_str(), fps_n, fps_d);
}

DslReturnType dsl_source_video_buffer_out_frame_rate_set(const wchar_t* name, 
    uint fps_n, uint fps_d)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceVideoBufferOutFrameRateSet(
        cstrName.c_str(), fps_n, fps_d);
}

DslReturnType dsl_source_video_buffer_out_crop_rectangle_get(const wchar_t* name,
    uint crop_at, uint* left, uint* top, uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(left);
    RETURN_IF_PARAM_IS_NULL(top);
    RETURN_IF_PARAM_IS_NULL(width);
    RETURN_IF_PARAM_IS_NULL(height);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceVideoBufferOutCropRectangleGet(
        cstrName.c_str(), crop_at, left, top, width, height);
}

DslReturnType dsl_source_video_buffer_out_crop_rectangle_set(const wchar_t* name,
    uint crop_at, uint left, uint top, uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceVideoBufferOutCropRectangleSet(
        cstrName.c_str(), crop_at, left, top, width, height);
}

DslReturnType dsl_source_video_buffer_out_orientation_get(const wchar_t* name,
    uint* orientation)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(orientation);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceVideoBufferOutOrientationGet(
        cstrName.c_str(), orientation);
}

DslReturnType dsl_source_video_buffer_out_orientation_set(const wchar_t* name,
    uint orientation)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceVideoBufferOutOrientationSet(
        cstrName.c_str(), orientation);
}

DslReturnType dsl_source_video_dewarper_add(const wchar_t* name, 
    const wchar_t* dewarper)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(dewarper);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrDewarper(dewarper);
    std::string cstrDewarper(wstrDewarper.begin(), wstrDewarper.end());

    return DSL::Services::GetServices()->SourceVideoDewarperAdd(cstrName.c_str(), 
        cstrDewarper.c_str());
}

DslReturnType dsl_source_video_dewarper_remove(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceVideoDewarperRemove(cstrName.c_str());
}

    
DslReturnType dsl_source_app_do_timestamp_get(const wchar_t* name, 
    boolean* do_timestamp)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(do_timestamp);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceAppDoTimestampGet(cstrName.c_str(), 
        do_timestamp);
}

DslReturnType dsl_source_app_do_timestamp_set(const wchar_t* name, 
    boolean do_timestamp)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceAppDoTimestampSet(cstrName.c_str(), 
        do_timestamp);
}
    
DslReturnType dsl_source_video_dimensions_get(const wchar_t* name, 
    uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(width);
    RETURN_IF_PARAM_IS_NULL(height);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceVideoDimensionsGet(cstrName.c_str(), 
        width, height);
}

DslReturnType dsl_source_frame_rate_get(const wchar_t* name, 
    uint* fps_n, uint* fps_d)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(fps_n);
    RETURN_IF_PARAM_IS_NULL(fps_d);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceFrameRateGet(cstrName.c_str(), 
        fps_n, fps_d);
}

DslReturnType dsl_source_uri_uri_get(const wchar_t* name, const wchar_t** uri)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(uri);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cUri;
    static std::string cstrUri;
    static std::wstring wcstrUri;
    
    uint retval = DSL::Services::GetServices()->SourceUriUriGet(cstrName.c_str(), 
        &cUri);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrUri.assign(cUri);
        wcstrUri.assign(cstrUri.begin(), cstrUri.end());
        *uri = wcstrUri.c_str();
    }
    return retval;
}

DslReturnType dsl_source_uri_uri_set(const wchar_t* name, const wchar_t* uri)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(uri);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUri(uri);
    std::string cstrUri(wstrUri.begin(), wstrUri.end());

    return DSL::Services::GetServices()->SourceUriUriSet(cstrName.c_str(), 
        cstrUri.c_str());
}

DslReturnType dsl_source_rtsp_uri_get(const wchar_t* name, const wchar_t** uri)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(uri);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cUri;
    static std::string cstrUri;
    static std::wstring wcstrUri;
    
    uint retval = DSL::Services::GetServices()->SourceRtspUriGet(cstrName.c_str(), 
        &cUri);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrUri.assign(cUri);
        wcstrUri.assign(cstrUri.begin(), cstrUri.end());
        *uri = wcstrUri.c_str();
    }
    return retval;
}

DslReturnType dsl_source_rtsp_uri_set(const wchar_t* name, const wchar_t* uri)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(uri);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUri(uri);
    std::string cstrUri(wstrUri.begin(), wstrUri.end());

    return DSL::Services::GetServices()->SourceRtspUriSet(cstrName.c_str(), 
        cstrUri.c_str());
}

DslReturnType dsl_source_rtsp_timeout_get(const wchar_t* name, uint* timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceRtspTimeoutGet(cstrName.c_str(), timeout);
}

DslReturnType dsl_source_rtsp_timeout_set(const wchar_t* name, uint timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceRtspTimeoutSet(cstrName.c_str(), timeout);
}

DslReturnType dsl_source_rtsp_connection_params_get(const wchar_t* name, 
    uint* sleep, uint* timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceRtspConnectionParamsGet(
        cstrName.c_str(), sleep, timeout);
}

DslReturnType dsl_source_rtsp_connection_params_set(const wchar_t* name, 
    uint sleep, uint timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceRtspConnectionParamsSet(
        cstrName.c_str(), sleep, timeout);
}

DslReturnType dsl_source_rtsp_connection_data_get(const wchar_t* name, dsl_rtsp_connection_data* data)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SourceRtspConnectionDataGet(
        cstrName.c_str(), data);
}

DslReturnType dsl_source_rtsp_connection_stats_clear(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceRtspConnectionStatsClear(
        cstrName.c_str());
}

DslReturnType dsl_source_rtsp_latency_get(const wchar_t* name, uint* latency)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(latency);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceRtspLatencyGet(
        cstrName.c_str(), latency);
}

DslReturnType dsl_source_rtsp_latency_set(const wchar_t* name, uint latency)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceRtspLatencySet(
        cstrName.c_str(), latency);
}

DslReturnType dsl_source_rtsp_drop_on_latency_enabled_get(const wchar_t* name, 
    boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceRtspDropOnLatencyEnabledGet(
        cstrName.c_str(), enabled);
}
    
DslReturnType dsl_source_rtsp_drop_on_latency_enabled_set(const wchar_t* name, 
    boolean enabled)    
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceRtspDropOnLatencyEnabledSet(
        cstrName.c_str(), enabled);
}

DslReturnType dsl_source_rtsp_tls_validation_flags_get(const wchar_t* name,
    uint* flags)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(flags);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceRtspTlsValidationFlagsGet(
        cstrName.c_str(), flags);
}
    
DslReturnType dsl_source_rtsp_tls_validation_flags_set(const wchar_t* name,
    uint flags)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceRtspTlsValidationFlagsSet(
        cstrName.c_str(), flags);
}
    
DslReturnType dsl_source_rtsp_state_change_listener_add(const wchar_t* source, 
    dsl_state_change_listener_cb listener, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(source);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrSource(source);
    std::string cstrSource(wstrSource.begin(), wstrSource.end());

    return DSL::Services::GetServices()->
        SourceRtspStateChangeListenerAdd(cstrSource.c_str(), listener, client_data);
}

DslReturnType dsl_source_rtsp_state_change_listener_remove(const wchar_t* source, 
    dsl_state_change_listener_cb listener)
{
    RETURN_IF_PARAM_IS_NULL(source);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrSource(source);
    std::string cstrSource(wstrSource.begin(), wstrSource.end());

    return DSL::Services::GetServices()->
        SourceRtspStateChangeListenerRemove(cstrSource.c_str(), listener);
}

DslReturnType dsl_source_rtsp_tap_add(const wchar_t* name, const wchar_t* tap)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(tap);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrTap(tap);
    std::string cstrTap(wstrTap.begin(), wstrTap.end());

    return DSL::Services::GetServices()->SourceRtspTapAdd(cstrName.c_str(), cstrTap.c_str());
}

DslReturnType dsl_source_rtsp_tap_remove(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceRtspTapRemove(cstrName.c_str());
}

DslReturnType dsl_source_unique_id_get(const wchar_t* name, int* unique_id)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(unique_id);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceUniqueIdGet(cstrName.c_str(), 
        unique_id);
}

DslReturnType dsl_source_stream_id_get(const wchar_t* name, int* stream_id)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(stream_id);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceStreamIdGet(cstrName.c_str(), 
        stream_id);
}

DslReturnType dsl_source_name_get(uint unique_id, const wchar_t** name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    const char* cName;
    static std::string cstrName;
    static std::wstring wcstrName;
    
    uint retval = DSL::Services::GetServices()->SourceNameGet(unique_id, &cName);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrName.assign(cName);
        wcstrName.assign(cstrName.begin(), cstrName.end());
        *name = wcstrName.c_str();
    }
    return retval;
}


DslReturnType dsl_source_pause(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourcePause(cstrName.c_str());
}

DslReturnType dsl_source_resume(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceResume(cstrName.c_str());
}

boolean dsl_source_is_live(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceIsLive(cstrName.c_str());
}

DslReturnType dsl_dewarper_new(const wchar_t* name, 
    const wchar_t* config_file, uint source_id)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());

    return DSL::Services::GetServices()->DewarperNew(cstrName.c_str(), 
        cstrConfig.c_str(), source_id);
}

DslReturnType dsl_dewarper_config_file_get(const wchar_t* name, 
    const wchar_t** config_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cConfig;
    static std::string cstrConfig;
    static std::wstring wcstrConfig;
    
    uint retval = DSL::Services::GetServices()->DewarperConfigFileGet(
        cstrName.c_str(), &cConfig);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrConfig.assign(cConfig);
        wcstrConfig.assign(cstrConfig.begin(), cstrConfig.end());
        *config_file = wcstrConfig.c_str();
    }
    return retval;
}

DslReturnType dsl_dewarper_config_file_set(const wchar_t* name, 
    const wchar_t* config_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(config_file);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());

    return DSL::Services::GetServices()->DewarperConfigFileSet(cstrName.c_str(), 
        cstrConfig.c_str());
}

DslReturnType dsl_dewarper_camera_id_get(const wchar_t* name, 
    uint* camera_id)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(camera_id);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DewarperCameraIdGet(
        cstrName.c_str(), camera_id);
}

DslReturnType dsl_dewarper_camera_id_set(const wchar_t* name, 
    uint camera_id)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DewarperCameraIdSet(
        cstrName.c_str(), camera_id);
}

DslReturnType dsl_dewarper_num_batch_buffers_get(
    const wchar_t* name, uint* num)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(num);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DewarperNumBatchBuffersGet(
        cstrName.c_str(), num);
}

DslReturnType dsl_dewarper_num_batch_buffers_set(
    const wchar_t* name, uint num)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DewarperNumBatchBuffersSet(
        cstrName.c_str(), num);
}

DslReturnType dsl_tap_record_new(const wchar_t* name, const wchar_t* outdir, 
     uint container, dsl_record_client_listener_cb client_listener)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(outdir);
    
    // Note: client_listener is optional in this case

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOutdir(outdir);
    std::string cstrOutdir(wstrOutdir.begin(), wstrOutdir.end());

    return DSL::Services::GetServices()->TapRecordNew(cstrName.c_str(), 
        cstrOutdir.c_str(), container, client_listener);
}     

DslReturnType dsl_tap_record_session_start(const wchar_t* name, 
     uint start, uint duration,void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TapRecordSessionStart(cstrName.c_str(), 
        start, duration, client_data);
}     

DslReturnType dsl_tap_record_session_stop(const wchar_t* name, 
    boolean sync)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TapRecordSessionStop(cstrName.c_str(), sync);
}

DslReturnType dsl_tap_record_outdir_get(const wchar_t* name, const wchar_t** outdir)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(outdir);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cOutdir;
    static std::string cstrOutdir;
    static std::wstring wcstrOutdir;
    
    uint retval = DSL::Services::GetServices()->TapRecordOutdirGet(cstrName.c_str(), &cOutdir);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrOutdir.assign(cOutdir);
        wcstrOutdir.assign(cstrOutdir.begin(), cstrOutdir.end());
        *outdir = wcstrOutdir.c_str();
    }
    return retval;
}

DslReturnType dsl_tap_record_outdir_set(const wchar_t* name, const wchar_t* outdir)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(outdir);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOutdir(outdir);
    std::string cstrOutdir(wstrOutdir.begin(), wstrOutdir.end());

    return DSL::Services::GetServices()->TapRecordOutdirSet(cstrName.c_str(), cstrOutdir.c_str());
}

DslReturnType dsl_tap_record_container_get(const wchar_t* name, uint* container)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TapRecordContainerGet(cstrName.c_str(), container);
}

DslReturnType dsl_tap_record_container_set(const wchar_t* name, uint container)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TapRecordContainerSet(cstrName.c_str(), container);
}
 
DslReturnType dsl_tap_record_cache_size_get(const wchar_t* name, uint* cache_size)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TapRecordCacheSizeGet(cstrName.c_str(), cache_size);
}

DslReturnType dsl_tap_record_cache_size_set(const wchar_t* name, uint cache_size)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TapRecordCacheSizeSet(cstrName.c_str(), cache_size);
}
 
DslReturnType dsl_tap_record_dimensions_get(const wchar_t* name, uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TapRecordDimensionsGet(cstrName.c_str(), width, height);
}

DslReturnType dsl_tap_record_dimensions_set(const wchar_t* name, uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TapRecordDimensionsSet(cstrName.c_str(), width, height);
}

DslReturnType dsl_tap_record_is_on_get(const wchar_t* name, boolean* is_on)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TapRecordIsOnGet(cstrName.c_str(), is_on);
}

DslReturnType dsl_tap_record_reset_done_get(const wchar_t* name, boolean* reset_done)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TapRecordResetDoneGet(cstrName.c_str(), reset_done);
}

DslReturnType dsl_tap_record_video_player_add(const wchar_t* name, 
    const wchar_t* player)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(player);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPlayer(player);
    std::string cstrPlayer(wstrPlayer.begin(), wstrPlayer.end());

    return DSL::Services::GetServices()->
        TapRecordVideoPlayerAdd(cstrName.c_str(), cstrPlayer.c_str());
}
    
DslReturnType dsl_tap_record_video_player_remove(const wchar_t* name, 
    const wchar_t* player)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(player);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPlayer(player);
    std::string cstrPlayer(wstrPlayer.begin(), wstrPlayer.end());

    return DSL::Services::GetServices()->
        TapRecordVideoPlayerRemove(cstrName.c_str(), cstrPlayer.c_str());
}

DslReturnType dsl_tap_record_mailer_add(const wchar_t* name, 
    const wchar_t* mailer, const wchar_t* subject)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(mailer);
    RETURN_IF_PARAM_IS_NULL(subject);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrMailer(mailer);
    std::string cstrMailer(wstrMailer.begin(), wstrMailer.end());
    std::wstring wstrSubject(subject);
    std::string cstrSubject(wstrSubject.begin(), wstrSubject.end());

    return DSL::Services::GetServices()->TapRecordMailerAdd(
        cstrName.c_str(), cstrMailer.c_str(), cstrSubject.c_str());
}
    
DslReturnType dsl_tap_record_mailer_remove(const wchar_t* name, 
    const wchar_t* mailer)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(mailer);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrMailer(mailer);
    std::string cstrMailer(wstrMailer.begin(), wstrMailer.end());

    return DSL::Services::GetServices()->TapRecordMailerRemove(
        cstrName.c_str(), cstrMailer.c_str());
}

DslReturnType dsl_preproc_new(const wchar_t* name, 
    const wchar_t* config_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(config_file);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfigFile(config_file);
    std::string cstrConfigFile(wstrConfigFile.begin(), wstrConfigFile.end());

    return DSL::Services::GetServices()->PreprocNew(
        cstrName.c_str(), cstrConfigFile.c_str());
}

DslReturnType dsl_preproc_config_file_get(const wchar_t* name, 
    const wchar_t** config_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cConfig;
    static std::string cstrConfig;
    static std::wstring wcstrConfig;
    
    uint retval = DSL::Services::GetServices()->PreprocConfigFileGet(cstrName.c_str(), 
        &cConfig);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrConfig.assign(cConfig);
        wcstrConfig.assign(cstrConfig.begin(), cstrConfig.end());
        *config_file = wcstrConfig.c_str();
    }
    return retval;
}

DslReturnType dsl_preproc_config_file_set(const wchar_t* name, 
    const wchar_t* config_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(config_file);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());

    return DSL::Services::GetServices()->PreprocConfigFileSet(cstrName.c_str(), 
        cstrConfig.c_str());
}

DslReturnType dsl_preproc_enabled_get(const wchar_t* name, 
    boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PreprocEnabledGet(cstrName.c_str(), 
        enabled);
}

DslReturnType dsl_preproc_enabled_set(const wchar_t* name, 
    boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PreprocEnabledSet(cstrName.c_str(), 
        enabled);
}

DslReturnType dsl_preproc_unique_id_get(const wchar_t* name, 
    uint* id)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(id);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PreprocUniqueIdGet(cstrName.c_str(), 
        id);
}

DslReturnType dsl_preproc_pph_add(const wchar_t* name, 
    const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->PreprocPphAdd(cstrName.c_str(), 
        cstrHandler.c_str(), pad);
}

DslReturnType dsl_preproc_pph_remove(const wchar_t* name,
    const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->PreprocPphRemove(cstrName.c_str(), 
        cstrHandler.c_str(), pad);
}

DslReturnType dsl_segvisual_new(const wchar_t* name, 
    uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SegVisualNew(cstrName.c_str(), 
        width, height);
}

DslReturnType dsl_segvisual_dimensions_get(const wchar_t* name, 
    uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SegVisualDimensionsGet(cstrName.c_str(), 
        width, height);
}

DslReturnType dsl_segvisual_dimensions_set(const wchar_t* name, 
    uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SegVisualDimensionsSet(cstrName.c_str(), 
        width, height);
}

DslReturnType dsl_segvisual_pph_add(const wchar_t* name, const wchar_t* handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->SegVisualPphAdd(cstrName.c_str(), 
        cstrHandler.c_str());
}

DslReturnType dsl_segvisual_pph_remove(const wchar_t* name, const wchar_t* handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->SegVisualPphRemove(cstrName.c_str(), 
        cstrHandler.c_str());
}

DslReturnType dsl_infer_gie_primary_new(const wchar_t* name, 
    const wchar_t* infer_config_file, const wchar_t* model_engine_file, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(infer_config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(infer_config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());
    
    std::string cstrEngine;
    if (model_engine_file != NULL)
    {
        std::wstring wstrEngine(model_engine_file);
        cstrEngine.assign(wstrEngine.begin(), wstrEngine.end());
    }
    return DSL::Services::GetServices()->InferPrimaryGieNew(cstrName.c_str(), 
        cstrConfig.c_str(), cstrEngine.c_str(), interval);
}

DslReturnType dsl_infer_tis_primary_new(const wchar_t* name, 
    const wchar_t* infer_config_file, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(infer_config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(infer_config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());
    
    return DSL::Services::GetServices()->InferPrimaryTisNew(cstrName.c_str(), 
        cstrConfig.c_str(), interval);
}

DslReturnType dsl_infer_gie_secondary_new(const wchar_t* name, 
    const wchar_t* infer_config_file, const wchar_t* model_engine_file, 
    const wchar_t* infer_on_gie, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(infer_config_file);
    RETURN_IF_PARAM_IS_NULL(infer_on_gie);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(infer_config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());
    std::wstring wstrInferOnGie(infer_on_gie);
    std::string cstrInferOnGie(wstrInferOnGie.begin(), wstrInferOnGie.end());

    std::string cstrEngine;
    if (model_engine_file != NULL)
    {
        std::wstring wstrEngine(model_engine_file);
        cstrEngine.assign(wstrEngine.begin(), wstrEngine.end());
    }
    return DSL::Services::GetServices()->InferSecondaryGieNew(cstrName.c_str(), 
        cstrConfig.c_str(), cstrEngine.c_str(), cstrInferOnGie.c_str(), interval);
}

DslReturnType dsl_infer_tis_secondary_new(const wchar_t* name, 
    const wchar_t* infer_config_file,
    const wchar_t* infer_on_tis, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(infer_config_file);
    RETURN_IF_PARAM_IS_NULL(infer_on_tis);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(infer_config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());
    std::wstring wstrInferOnTis(infer_on_tis);
    std::string cstrInferOnTis(wstrInferOnTis.begin(), wstrInferOnTis.end());

    return DSL::Services::GetServices()->InferSecondaryTisNew(cstrName.c_str(), 
        cstrConfig.c_str(), cstrInferOnTis.c_str(), interval);
}

DslReturnType dsl_infer_batch_size_get(const wchar_t* name, uint* size)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(size);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->InferBatchSizeGet(cstrName.c_str(), size);
}

DslReturnType dsl_infer_batch_size_set(const wchar_t* name, uint size)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->InferBatchSizeSet(cstrName.c_str(), size);
}

DslReturnType dsl_infer_unique_id_get(const wchar_t* name, uint* id)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->InferUniqueIdGet(cstrName.c_str(), id);
}

DslReturnType dsl_infer_pph_add(const wchar_t* name, 
    const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->InferPphAdd(cstrName.c_str(), 
        cstrHandler.c_str(), pad);
}

DslReturnType dsl_infer_pph_remove(const wchar_t* name,
    const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->InferPphRemove(cstrName.c_str(), 
        cstrHandler.c_str(), pad);
}

DslReturnType dsl_infer_config_file_get(const wchar_t* name, 
    const wchar_t** infer_config_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(infer_config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cConfig;
    static std::string cstrConfig;
    static std::wstring wcstrConfig;
    
    uint retval = DSL::Services::GetServices()->InferConfigFileGet(
        cstrName.c_str(), &cConfig);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrConfig.assign(cConfig);
        wcstrConfig.assign(cstrConfig.begin(), cstrConfig.end());
        *infer_config_file = wcstrConfig.c_str();
    }
    return retval;
}

DslReturnType dsl_infer_config_file_set(const wchar_t* name, 
    const wchar_t* infer_config_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(infer_config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(infer_config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());

    return DSL::Services::GetServices()->InferConfigFileSet(cstrName.c_str(), 
        cstrConfig.c_str());
}

DslReturnType dsl_infer_gie_model_engine_file_get(const wchar_t* name, 
    const wchar_t** model_engine_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(model_engine_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cEngine;
    static std::string cstrEngine;
    static std::wstring wcstrEngine;
    
    uint retval = DSL::Services::GetServices()->InferGieModelEngineFileGet(
        cstrName.c_str(), &cEngine);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrEngine.assign(cEngine);
        wcstrEngine.assign(cstrEngine.begin(), cstrEngine.end());
        *model_engine_file = wcstrEngine.c_str();
    }
    return retval;
}
 
DslReturnType dsl_infer_gie_model_engine_file_set(const wchar_t* name, 
    const wchar_t* model_engine_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(model_engine_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrEngine(model_engine_file);
    std::string cstrEngine(wstrEngine.begin(), wstrEngine.end());

    return DSL::Services::GetServices()->InferGieModelEngineFileSet(
        cstrName.c_str(), cstrEngine.c_str());
}

DslReturnType dsl_infer_gie_tensor_meta_settings_get(const wchar_t* name, 
    boolean* input_enabled, boolean* output_enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(input_enabled);
    RETURN_IF_PARAM_IS_NULL(output_enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->InferGieTensorMetaSettingsGet(
        cstrName.c_str(), input_enabled, output_enabled);
}
    
DslReturnType dsl_infer_gie_tensor_meta_settings_set(const wchar_t* name, 
    boolean input_enabled, boolean output_enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->InferGieTensorMetaSettingsSet(
        cstrName.c_str(), input_enabled, output_enabled);
}
    
DslReturnType dsl_infer_interval_get(const wchar_t* name, uint* interval)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(interval);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->InferIntervalGet(cstrName.c_str(), interval);
}

DslReturnType dsl_infer_interval_set(const wchar_t* name, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->InferIntervalSet(cstrName.c_str(), interval);
}


DslReturnType dsl_infer_raw_output_enabled_set(const wchar_t* name, 
    boolean enabled, const wchar_t* path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPath(path);
    std::string cstrPath(wstrPath.begin(), wstrPath.end());

    return DSL::Services::GetServices()->InferRawOutputEnabledSet(cstrName.c_str(), 
        enabled, cstrPath.c_str());
}

DslReturnType dsl_tracker_new(const wchar_t* name, 
    const wchar_t* config_file, uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrCfgFile;
    if (config_file != NULL)
    {
        std::wstring wstrCfgFile(config_file);
        cstrCfgFile.assign(wstrCfgFile.begin(), wstrCfgFile.end());
    }

    return DSL::Services::GetServices()->TrackerNew(cstrName.c_str(), 
        cstrCfgFile.c_str(), width, height);
}

DslReturnType dsl_tracker_lib_file_get(const wchar_t* name, 
    const wchar_t** lib_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(lib_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cLib;
    static std::string cstrLib;
    static std::wstring wcstrLib;
    
    uint retval = DSL::Services::GetServices()->TrackerLibFileGet(
        cstrName.c_str(), &cLib);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrLib.assign(cLib);
        wcstrLib.assign(cstrLib.begin(), cstrLib.end());
        *lib_file = wcstrLib.c_str();
    }
    return retval;
}

DslReturnType dsl_tracker_lib_file_set(const wchar_t* name, 
    const wchar_t* lib_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(lib_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrLib(lib_file);
    std::string cstrLib(wstrLib.begin(), wstrLib.end());

    return DSL::Services::GetServices()->TrackerLibFileSet(
        cstrName.c_str(), cstrLib.c_str());
}

DslReturnType dsl_tracker_config_file_get(const wchar_t* name, 
    const wchar_t** config_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cConfig;
    static std::string cstrConfig;
    static std::wstring wcstrConfig;
    
    uint retval = DSL::Services::GetServices()->TrackerConfigFileGet(
        cstrName.c_str(), &cConfig);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrConfig.assign(cConfig);
        wcstrConfig.assign(cstrConfig.begin(), cstrConfig.end());
        *config_file = wcstrConfig.c_str();
    }
    return retval;
}

DslReturnType dsl_tracker_config_file_set(const wchar_t* name, 
    const wchar_t* config_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());

    return DSL::Services::GetServices()->TrackerConfigFileSet(
        cstrName.c_str(), cstrConfig.c_str());
}

DslReturnType dsl_tracker_dimensions_get(const wchar_t* name, 
    uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(width);
    RETURN_IF_PARAM_IS_NULL(height);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TrackerDimensionsGet(cstrName.c_str(), 
        width, height);
}

DslReturnType dsl_tracker_dimensions_set(const wchar_t* name, 
    uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TrackerDimensionsSet(cstrName.c_str(), 
        width, height);
}

DslReturnType dsl_tracker_tensor_meta_settings_get(const wchar_t* name, 
    boolean* input_enabled, const wchar_t** track_on_gie)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(input_enabled);
    RETURN_IF_PARAM_IS_NULL(track_on_gie);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cTrackOnGie;
    static std::string cstrTrackOnGie;
    static std::wstring wcstrTrackOnGie;
    
    uint retval = DSL::Services::GetServices()->TrackerTensorMetaSettingsGet(
        cstrName.c_str(), input_enabled, &cTrackOnGie);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrTrackOnGie.assign(cTrackOnGie);
        wcstrTrackOnGie.assign(cstrTrackOnGie.begin(), cstrTrackOnGie.end());
        *track_on_gie = wcstrTrackOnGie.c_str();
    }
    return retval;
}

DslReturnType dsl_tracker_tensor_meta_settings_set(const wchar_t* name, 
    boolean input_enabled, const wchar_t* track_on_gie)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    std::string cstrTrackOnGie;
    if (track_on_gie)
    {
        std::wstring wstrTrackOnGie(track_on_gie);
        cstrTrackOnGie.assign(wstrTrackOnGie.begin(), wstrTrackOnGie.end());
    }

    return DSL::Services::GetServices()->TrackerTensorMetaSettingsSet(
        cstrName.c_str(), input_enabled, cstrTrackOnGie.c_str());
}
    
DslReturnType dsl_tracker_id_display_enabled_get(const wchar_t* name, 
    boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TrackerIdDisplayEnabledGet(
        cstrName.c_str(), enabled);
}
    
DslReturnType dsl_tracker_id_display_enabled_set(const wchar_t* name, 
    boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TrackerIdDisplayEnabledSet(
        cstrName.c_str(), enabled);
}
    
DslReturnType dsl_tracker_pph_add(const wchar_t* name,
    const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->TrackerPphAdd(cstrName.c_str(), 
        cstrHandler.c_str(), pad);
}

DslReturnType dsl_tracker_pph_remove(const wchar_t* name,
    const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->TrackerPphRemove(cstrName.c_str(), 
        cstrHandler.c_str(), pad);
}

DslReturnType dsl_ofv_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OfvNew(cstrName.c_str());
}

DslReturnType dsl_osd_new(const wchar_t* name, 
    boolean text_enabled, boolean clock_enabled, 
    boolean bbox_enabled, boolean mask_enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdNew(cstrName.c_str(), 
        text_enabled, clock_enabled, bbox_enabled, mask_enabled);
}

DslReturnType dsl_osd_text_enabled_get(const wchar_t* name, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdTextEnabledGet(cstrName.c_str(), enabled);
}

DslReturnType dsl_osd_text_enabled_set(const wchar_t* name, boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdTextEnabledSet(cstrName.c_str(), enabled);
}

DslReturnType dsl_osd_clock_enabled_get(const wchar_t* name, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockEnabledGet(cstrName.c_str(), enabled);
}

DslReturnType dsl_osd_clock_enabled_set(const wchar_t* name, boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockEnabledSet(cstrName.c_str(), enabled);
}

DslReturnType dsl_osd_clock_offsets_get(const wchar_t* name, 
    uint* offset_x, uint* offset_y)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(offset_x);
    RETURN_IF_PARAM_IS_NULL(offset_y);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockOffsetsGet(cstrName.c_str(), 
        offset_x, offset_y);
}

DslReturnType dsl_osd_clock_offsets_set(const wchar_t* name, 
    uint offset_x, uint offset_y)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockOffsetsSet(cstrName.c_str(), 
        offset_x, offset_y);
}

DslReturnType dsl_osd_clock_font_get(const wchar_t* name, 
    const wchar_t** font, uint* size)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(font);
    RETURN_IF_PARAM_IS_NULL(size);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cfont;
    static std::string cstrFont;
    static std::wstring wcstrFont;
    
    uint retval = DSL::Services::GetServices()->OsdClockFontGet(cstrName.c_str(), 
        &cfont, size);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrFont.assign(cfont);
        wcstrFont.assign(cstrFont.begin(), cstrFont.end());
        *font = wcstrFont.c_str();
    }
    return retval;
}

DslReturnType dsl_osd_clock_font_set(const wchar_t* name, 
    const wchar_t* font, uint size)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(font);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFont(font);
    std::string cstrFont(wstrFont.begin(), wstrFont.end());

    return DSL::Services::GetServices()->OsdClockFontSet(cstrName.c_str(), 
        cstrFont.c_str(), size);
}

DslReturnType dsl_osd_clock_color_get(const wchar_t* name, 
    double* red, double* green, double* blue, double* alpha)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(red);
    RETURN_IF_PARAM_IS_NULL(green);
    RETURN_IF_PARAM_IS_NULL(blue);
    RETURN_IF_PARAM_IS_NULL(alpha);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockColorGet(cstrName.c_str(), 
        red, green, blue, alpha);
}

DslReturnType dsl_osd_clock_color_set(const wchar_t* 
    name, double red, double green, double blue, double alpha)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockColorSet(cstrName.c_str(), 
        red, green, blue, alpha);
}

DslReturnType dsl_osd_bbox_enabled_get(const wchar_t* name, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdBboxEnabledGet(cstrName.c_str(), 
        enabled);
}

DslReturnType dsl_osd_bbox_enabled_set(const wchar_t* name, boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdBboxEnabledSet(cstrName.c_str(), 
        enabled);
}

DslReturnType dsl_osd_mask_enabled_get(const wchar_t* name, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdMaskEnabledGet(cstrName.c_str(), 
        enabled);
}

DslReturnType dsl_osd_mask_enabled_set(const wchar_t* name, boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdMaskEnabledSet(cstrName.c_str(), 
        enabled);
}

DslReturnType dsl_osd_process_mode_get(const wchar_t* name, uint* mode)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(mode);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdProcessModeGet(cstrName.c_str(), 
        mode);
}

DslReturnType dsl_osd_process_mode_set(const wchar_t* name, uint mode)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdProcessModeSet(cstrName.c_str(), 
        mode);
}

DslReturnType dsl_osd_pph_add(const wchar_t* name,
    const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->OsdPphAdd(cstrName.c_str(), 
        cstrHandler.c_str(), pad);
}

DslReturnType dsl_osd_pph_remove(const wchar_t* name,
    const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->OsdPphRemove(cstrName.c_str(), 
        cstrHandler.c_str(), pad);
}

DslReturnType dsl_tee_demuxer_new(const wchar_t* name, 
    uint max_branches)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TeeDemuxerNew(cstrName.c_str(),
        max_branches);
}

DslReturnType dsl_tee_demuxer_new_branch_add_many(const wchar_t* name, 
    uint max_branches, const wchar_t** branches)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branches);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    DslReturnType retval = DSL::Services::GetServices()->TeeDemuxerNew(
        cstrName.c_str(), max_branches);
    if (retval != DSL_RESULT_SUCCESS)
    {
        return retval;
    }

    for (const wchar_t** branch = branches; *branch; branch++)
    {
        std::wstring wstrBranch(*branch);
        std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        retval = DSL::Services::GetServices()->TeeBranchAdd(cstrName.c_str(), 
            cstrBranch.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_tee_demuxer_branch_add_to(const wchar_t* name, 
    const wchar_t* branch, uint stream_id)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branch);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());

    return DSL::Services::GetServices()->TeeDemuxerBranchAddTo(cstrName.c_str(),
        cstrBranch.c_str(), stream_id);
}
    
DslReturnType dsl_tee_demuxer_branch_move_to(const wchar_t* name, 
    const wchar_t* branch, uint stream_id)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branch);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());

    return DSL::Services::GetServices()->TeeDemuxerBranchMoveTo(cstrName.c_str(),
        cstrBranch.c_str(), stream_id);
}
    
DslReturnType dsl_tee_demuxer_max_branches_get(const wchar_t* name, 
    uint* max_branches)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(max_branches);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TeeDemuxerMaxBranchesGet(cstrName.c_str(),
        max_branches);
}
    
DslReturnType dsl_tee_demuxer_max_branches_set(const wchar_t* name, 
    uint max_branches)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TeeDemuxerMaxBranchesSet(
        cstrName.c_str(), max_branches);
}

DslReturnType dsl_tee_splitter_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TeeSplitterNew(cstrName.c_str());
}

DslReturnType dsl_tee_splitter_new_branch_add_many(const wchar_t* name, 
    const wchar_t** branches)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branches);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    DslReturnType retval = DSL::Services::GetServices()->TeeSplitterNew(
        cstrName.c_str());
    if (retval != DSL_RESULT_SUCCESS)
    {
        return retval;
    }

    for (const wchar_t** branch = branches; *branch; branch++)
    {
        std::wstring wstrBranch(*branch);
        std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        retval = DSL::Services::GetServices()->TeeBranchAdd(cstrName.c_str(), 
            cstrBranch.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_tee_branch_add(const wchar_t* name, const wchar_t* branch)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branch);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());

    return DSL::Services::GetServices()->TeeBranchAdd(cstrName.c_str(), 
        cstrBranch.c_str());
}

DslReturnType dsl_tee_branch_add_many(const wchar_t* name, 
    const wchar_t** branches)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branches);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    for (const wchar_t** branch = branches; *branch; branch++)
    {
        std::wstring wstrBranch(*branch);
        std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        DslReturnType retval = DSL::Services::GetServices()->
            TeeBranchAdd(cstrName.c_str(), cstrBranch.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_tee_branch_remove(const wchar_t* name, const wchar_t* branch)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branch);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());

    return DSL::Services::GetServices()->TeeBranchRemove(cstrName.c_str(), 
        cstrBranch.c_str());
}

DslReturnType dsl_tee_branch_remove_many(const wchar_t* name, 
    const wchar_t** branches)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branches);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    for (const wchar_t** branch = branches; *branch; branch++)
    {
        std::wstring wstrBranch(*branch);
        std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        DslReturnType retval = DSL::Services::GetServices()->
            TeeBranchRemove(cstrName.c_str(), cstrBranch.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_tee_branch_remove_all(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TeeBranchRemoveAll(cstrName.c_str());
}

DslReturnType dsl_tee_branch_count_get(const wchar_t* name, uint* count)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TeeBranchCountGet(cstrName.c_str(), count);
}

DslReturnType dsl_tee_blocking_timeout_get(const wchar_t* name, 
    uint* timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(timeout);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TeeBlockingTimeoutGet(
        cstrName.c_str(), timeout);
}

DslReturnType dsl_tee_blocking_timeout_set(const wchar_t* name, 
    uint timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TeeBlockingTimeoutSet(
        cstrName.c_str(), timeout);
}
    
DslReturnType dsl_tee_pph_add(const wchar_t* name, const wchar_t* handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->TeePphAdd(cstrName.c_str(), 
        cstrHandler.c_str());
}

DslReturnType dsl_tee_pph_remove(const wchar_t* name, const wchar_t* handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->TeePphRemove(cstrName.c_str(), 
        cstrHandler.c_str());
}

DslReturnType dsl_remuxer_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->RemuxerNew(cstrName.c_str());
}

DslReturnType dsl_remuxer_new_branch_add_many(const wchar_t* name, 
    const wchar_t** branches)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branches);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    DslReturnType retval = DSL::Services::GetServices()->RemuxerNew(
        cstrName.c_str());
    if (retval != DSL_RESULT_SUCCESS)
    {
        return retval;
    }

    for (const wchar_t** branch = branches; *branch; branch++)
    {
        std::wstring wstrBranch(*branch);
        std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        retval = DSL::Services::GetServices()->RemuxerBranchAdd(cstrName.c_str(), 
            cstrBranch.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_remuxer_branch_add_to(const wchar_t* name, 
    const wchar_t* branch, uint* stream_ids, uint num_stream_ids)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branch);
    RETURN_IF_PARAM_IS_NULL(stream_ids);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());

    return DSL::Services::GetServices()->RemuxerBranchAddTo(cstrName.c_str(), 
        cstrBranch.c_str(), stream_ids, num_stream_ids);
}

DslReturnType dsl_remuxer_branch_add(const wchar_t* name, const wchar_t* branch)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branch);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());

    return DSL::Services::GetServices()->RemuxerBranchAdd(cstrName.c_str(), 
        cstrBranch.c_str());
}

DslReturnType dsl_remuxer_branch_add_many(const wchar_t* name, 
    const wchar_t** branches)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branches);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    for (const wchar_t** branch = branches; *branch; branch++)
    {
        std::wstring wstrBranch(*branch);
        std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        DslReturnType retval = DSL::Services::GetServices()->
            RemuxerBranchAdd(cstrName.c_str(), cstrBranch.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_remuxer_branch_remove(const wchar_t* name, const wchar_t* branch)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branch);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());

    return DSL::Services::GetServices()->RemuxerBranchRemove(cstrName.c_str(), 
        cstrBranch.c_str());
}

DslReturnType dsl_remuxer_branch_remove_many(const wchar_t* name, 
    const wchar_t** branches)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branches);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    for (const wchar_t** branch = branches; *branch; branch++)
    {
        std::wstring wstrBranch(*branch);
        std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        DslReturnType retval = DSL::Services::GetServices()->
            RemuxerBranchRemove(cstrName.c_str(), cstrBranch.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_remuxer_branch_remove_all(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->RemuxerBranchRemoveAll(cstrName.c_str());
}

DslReturnType dsl_remuxer_branch_count_get(const wchar_t* name, uint* count)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->RemuxerBranchCountGet(cstrName.c_str(), count);
}
DslReturnType dsl_remuxer_batch_size_get(const wchar_t* name, 
    uint* batch_size)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(batch_size);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->RemuxerBatchSizeGet(
        cstrName.c_str(), batch_size);
}

DslReturnType dsl_remuxer_batch_size_set(const wchar_t* name, 
    uint batch_size)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->RemuxerBatchSizeSet(
        cstrName.c_str(), batch_size);
}

// -----------------------------------------------------------------------------------
// NEW STREAMMUX SERVICES - End

DslReturnType dsl_remuxer_branch_config_file_get(const wchar_t* name, 
    const wchar_t* branch, const wchar_t** config_file)
{
    RETURN_IF_NEW_NVSTREAMMUX_DISABLED();
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branch);
    RETURN_IF_PARAM_IS_NULL(config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
    
    const char* cConfig;
    static std::string cstrConfig;
    static std::wstring wcstrConfig;
    
    uint retval = DSL::Services::GetServices()->RemuxerBranchConfigFileGet(
        cstrName.c_str(), cstrBranch.c_str(), &cConfig);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrConfig.assign(cConfig);
        wcstrConfig.assign(cstrConfig.begin(), cstrConfig.end());
        *config_file = wcstrConfig.c_str();
    }
    return retval;
}
    
DslReturnType dsl_remuxer_branch_config_file_set(const wchar_t* name, 
    const wchar_t* branch, const wchar_t* config_file)
{
    RETURN_IF_NEW_NVSTREAMMUX_DISABLED();
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branch);
    RETURN_IF_PARAM_IS_NULL(config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        std::wstring wstrConfig(config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());

    return DSL::Services::GetServices()->RemuxerBranchConfigFileSet(
        cstrName.c_str(), cstrBranch.c_str(), cstrConfig.c_str());
}

// -----------------------------------------------------------------------------------
// NEW STREAMMUX SERVICES - End
// -----------------------------------------------------------------------------------
// OLD STREAMMUX SERVICES - Start

DslReturnType dsl_remuxer_batch_properties_get(const wchar_t* name, 
    uint* batch_size, int* batch_timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(batch_size);
    RETURN_IF_PARAM_IS_NULL(batch_timeout);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->RemuxerBatchPropertiesGet(
        cstrName.c_str(), batch_size, batch_timeout);
}

DslReturnType dsl_remuxer_batch_properties_set(const wchar_t* name, 
    uint batch_size, int batch_timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->RemuxerBatchPropertiesSet(
        cstrName.c_str(), batch_size, batch_timeout);
}

DslReturnType dsl_remuxer_dimensions_get(const wchar_t* name, 
    uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(width);
    RETURN_IF_PARAM_IS_NULL(height);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->RemuxerDimensionsGet(
        cstrName.c_str(), width, height);
}

DslReturnType dsl_remuxer_dimensions_set(const wchar_t* name, 
    uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->RemuxerDimensionsSet(
        cstrName.c_str(), width, height);
}
    
// -----------------------------------------------------------------------------------
// OLD STREAMMUX SERVICES - End

DslReturnType dsl_remuxer_pph_add(const wchar_t* name, 
    const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->RemuxerPphAdd(cstrName.c_str(), 
        cstrHandler.c_str(), pad);
}

DslReturnType dsl_remuxer_pph_remove(const wchar_t* name, 
    const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->RemuxerPphRemove(cstrName.c_str(), 
        cstrHandler.c_str(), pad);
}

DslReturnType dsl_tiler_new(const wchar_t* name, uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TilerNew(cstrName.c_str(), width, height);
}

DslReturnType dsl_tiler_dimensions_get(const wchar_t* name, uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(width);
    RETURN_IF_PARAM_IS_NULL(height);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TilerDimensionsGet(cstrName.c_str(), width, height);
}

DslReturnType dsl_tiler_dimensions_set(const wchar_t* name, uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TilerDimensionsSet(cstrName.c_str(), width, height);
}

DslReturnType dsl_tiler_tiles_get(const wchar_t* name, uint* cols, uint* rows)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(cols);
    RETURN_IF_PARAM_IS_NULL(rows);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TilerTilesGet(cstrName.c_str(), cols, rows);
}

DslReturnType dsl_tiler_tiles_set(const wchar_t* name, uint cols, uint rows)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TilerTilesSet(cstrName.c_str(), cols, rows);
}

DslReturnType dsl_tiler_frame_numbering_enabled_get(const wchar_t* name,
    boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TilerFrameNumberingEnabledGet(
        cstrName.c_str(), enabled);
}
    
DslReturnType dsl_tiler_frame_numbering_enabled_set(const wchar_t* name,
    boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TilerFrameNumberingEnabledSet(
        cstrName.c_str(), enabled);
}
    
DslReturnType dsl_tiler_source_show_get(const wchar_t* name, 
    const wchar_t** source, uint* timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(source);
    RETURN_IF_PARAM_IS_NULL(timeout);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cSource;
    static std::string cstrSource;
    static std::wstring wcstrSource;
    
    uint retval = DSL::Services::GetServices()->TilerSourceShowGet(cstrName.c_str(), &cSource, timeout);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        if (cSource == NULL)
        {
            *source = NULL;
        }
        else
        {
            cstrSource.assign(cSource);
            wcstrSource.assign(cstrSource.begin(), cstrSource.end());
            *source = wcstrSource.c_str();
        }
    }
    return retval;
}

DslReturnType dsl_tiler_source_show_set(const wchar_t* name, 
    const wchar_t* source, uint timeout, boolean has_precedence)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(source);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrSource(source);
    std::string cstrSource(wstrSource.begin(), wstrSource.end());

    return DSL::Services::GetServices()->TilerSourceShowSet(cstrName.c_str(), 
        cstrSource.c_str(), timeout, has_precedence);
}

DslReturnType dsl_tiler_source_show_select(const wchar_t* name, 
    int x_pos, int y_pos, uint window_width, uint window_height, uint timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    return DSL::Services::GetServices()->TilerSourceShowSelect(cstrName.c_str(),
        x_pos, y_pos, window_width, window_height, timeout);
}

DslReturnType dsl_tiler_source_show_all(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    return DSL::Services::GetServices()->TilerSourceShowAll(cstrName.c_str());
}

DslReturnType dsl_tiler_source_show_cycle(const wchar_t* name, uint timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    return DSL::Services::GetServices()->TilerSourceShowCycle(cstrName.c_str(), timeout);
}

DslReturnType dsl_tiler_pph_add(const wchar_t* name, const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());

    return DSL::Services::GetServices()->TilerPphAdd(cstrName.c_str(), cstrHandler.c_str(), pad);
}     

DslReturnType dsl_tiler_pph_remove(const wchar_t* name, const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());

    return DSL::Services::GetServices()->TilerPphRemove(cstrName.c_str(), cstrHandler.c_str(), pad);
}     

DslReturnType dsl_sink_app_new(const wchar_t* name, uint data_type,
    dsl_sink_app_new_data_handler_cb client_handler, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(client_handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkAppNew(cstrName.c_str(),
        data_type, client_handler, client_data);
}

DslReturnType dsl_sink_app_data_type_get(const wchar_t* name, uint* data_type)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(data_type);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkAppDataTypeGet(cstrName.c_str(),
        data_type);
}
    
DslReturnType dsl_sink_app_data_type_set(const wchar_t* name, uint data_type)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkAppDataTypeSet(cstrName.c_str(),
        data_type);
}
    
DslReturnType dsl_sink_fake_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkFakeNew(cstrName.c_str());
}

DslReturnType dsl_sink_window_3d_new(const wchar_t* name, 
    uint offset_x, uint offset_y, uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindow3dNew(cstrName.c_str(), 
        offset_x, offset_y, width, height);
}

DslReturnType dsl_sink_window_egl_new(const wchar_t* name,
    uint offset_x, uint offset_y, uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowEglNew(cstrName.c_str(), 
        offset_x, offset_y, width, height);
}

DslReturnType dsl_sink_window_offsets_get(const wchar_t* name, 
    uint* offset_x, uint* offset_y)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(offset_x);
    RETURN_IF_PARAM_IS_NULL(offset_y);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowOffsetsGet(
        cstrName.c_str(), offset_x, offset_y);
}

DslReturnType dsl_sink_window_offsets_set(const wchar_t* name, 
    uint offset_x, uint offset_y)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowOffsetsSet(
        cstrName.c_str(), offset_x, offset_y);
}

DslReturnType dsl_sink_window_dimensions_get(const wchar_t* name, 
    uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(width);
    RETURN_IF_PARAM_IS_NULL(height);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowDimensionsGet(
        cstrName.c_str(), width, height);
}

DslReturnType dsl_sink_window_dimensions_set(const wchar_t* name, 
    uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowDimensionsSet(
        cstrName.c_str(), width, height);
}

DslReturnType dsl_sink_window_handle_get(const wchar_t* name, uint64_t* handle)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handle);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowHandleGet(
        cstrName.c_str(), handle);
}

DslReturnType dsl_sink_window_handle_set(const wchar_t* name, uint64_t handle)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowHandleSet(
        cstrName.c_str(), handle);
}

DslReturnType dsl_sink_window_clear(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowClear(cstrName.c_str());
}
 
DslReturnType dsl_sink_window_egl_force_aspect_ratio_get(const wchar_t* name, 
    boolean* force)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(force);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowEglForceAspectRatioGet(
        cstrName.c_str(), force);
}
    
DslReturnType dsl_sink_window_egl_force_aspect_ratio_set(const wchar_t* name, 
    boolean force)    
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowEglForceAspectRatioSet(
        cstrName.c_str(), force);
}

DslReturnType dsl_sink_window_fullscreen_enabled_get(const wchar_t* name,
    boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowFullScreenEnabledGet(
        cstrName.c_str(), enabled);
}

DslReturnType dsl_sink_window_fullscreen_enabled_set(const wchar_t* name, 
    boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowFullScreenEnabledSet(
        cstrName.c_str(), enabled);
}

DslReturnType dsl_sink_window_key_event_handler_add(const wchar_t* name, 
    dsl_sink_window_key_event_handler_cb handler, void* client_data)
{
    std::cout << "handler = " << handler << std::endl;
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowKeyEventHandlerAdd(
        cstrName.c_str(), handler, client_data);
}    

DslReturnType dsl_sink_window_key_event_handler_remove(const wchar_t* name, 
    dsl_sink_window_key_event_handler_cb handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowKeyEventHandlerRemove(
        cstrName.c_str(), handler);
}

DslReturnType dsl_sink_window_button_event_handler_add(const wchar_t* name, 
    dsl_sink_window_button_event_handler_cb handler, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowButtonEventHandlerAdd(
        cstrName.c_str(), handler, client_data);
}    

DslReturnType dsl_sink_window_button_event_handler_remove(const wchar_t* name, 
    dsl_sink_window_button_event_handler_cb handler)    
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowButtonEventHandlerRemove(
        cstrName.c_str(), handler);
}

DslReturnType dsl_sink_window_delete_event_handler_add(const wchar_t* name, 
    dsl_sink_window_delete_event_handler_cb handler, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowDeleteEventHandlerAdd(
        cstrName.c_str(), handler, client_data);
}    

DslReturnType dsl_sink_window_delete_event_handler_remove(const wchar_t* name, 
    dsl_sink_window_delete_event_handler_cb handler)    
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        SinkWindowDeleteEventHandlerRemove(cstrName.c_str(), handler);
}

DslReturnType dsl_sink_file_new(const wchar_t* name, const wchar_t* file_path, 
     uint codec, uint container, uint bitrate, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPath(file_path);
    std::string cstrPath(wstrPath.begin(), wstrPath.end());

    return DSL::Services::GetServices()->SinkFileNew(cstrName.c_str(), 
        cstrPath.c_str(), codec, container, bitrate, interval);
}     

DslReturnType dsl_sink_encode_settings_get(const wchar_t* name,
    uint* codec, uint* bitrate, uint* interval)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(codec);
    RETURN_IF_PARAM_IS_NULL(bitrate);
    RETURN_IF_PARAM_IS_NULL(interval);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkEncodeSettingsGet(cstrName.c_str(), 
        codec, bitrate, interval);
}    

DslReturnType dsl_sink_encode_settings_set(const wchar_t* name,
    uint codec, uint bitrate, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkEncodeSettingsSet(cstrName.c_str(), 
        codec, bitrate, interval);
}

DslReturnType dsl_sink_encode_dimensions_get(const wchar_t* name, 
    uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(width);
    RETURN_IF_PARAM_IS_NULL(height);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkEncodeDimensionsGet(cstrName.c_str(), 
        width, height);
}

DslReturnType dsl_sink_encode_dimensions_set(const wchar_t* name, 
    uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkEncodeDimensionsSet(cstrName.c_str(), 
        width, height);
}

DslReturnType dsl_sink_record_new(const wchar_t* name, const wchar_t* outdir, 
     uint codec, uint container, uint bitrate, uint interval, dsl_record_client_listener_cb client_listener)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(outdir);

    //Note client_listener is optional in the case.

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOutdir(outdir);
    std::string cstrOutdir(wstrOutdir.begin(), wstrOutdir.end());

    return DSL::Services::GetServices()->SinkRecordNew(cstrName.c_str(), 
        cstrOutdir.c_str(), codec, container, bitrate, interval, client_listener);
}     

DslReturnType dsl_sink_record_session_start(const wchar_t* name, 
     uint start, uint duration,void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordSessionStart(cstrName.c_str(), 
        start, duration, client_data);
}     

DslReturnType dsl_sink_record_session_stop(const wchar_t* name, 
    boolean sync)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordSessionStop(cstrName.c_str(), sync);
}

DslReturnType dsl_sink_record_outdir_get(const wchar_t* name, const wchar_t** outdir)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(outdir);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cOutdir;
    static std::string cstrOutdir;
    static std::wstring wcstrOutdir;
    
    uint retval = DSL::Services::GetServices()->SinkRecordOutdirGet(cstrName.c_str(), &cOutdir);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrOutdir.assign(cOutdir);
        wcstrOutdir.assign(cstrOutdir.begin(), cstrOutdir.end());
        *outdir = wcstrOutdir.c_str();
    }
    return retval;
}

DslReturnType dsl_sink_record_outdir_set(const wchar_t* name, const wchar_t* outdir)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(outdir);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOutdir(outdir);
    std::string cstrOutdir(wstrOutdir.begin(), wstrOutdir.end());

    return DSL::Services::GetServices()->SinkRecordOutdirSet(cstrName.c_str(), cstrOutdir.c_str());
}

DslReturnType dsl_sink_record_container_get(const wchar_t* name, uint* container)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordContainerGet(cstrName.c_str(), container);
}

DslReturnType dsl_sink_record_container_set(const wchar_t* name, uint container)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordContainerSet(cstrName.c_str(), container);
}

DslReturnType dsl_sink_record_cache_size_get(const wchar_t* name, uint* cache_size)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordCacheSizeGet(cstrName.c_str(), cache_size);
}

DslReturnType dsl_sink_record_cache_size_set(const wchar_t* name, uint cache_size)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordCacheSizeSet(cstrName.c_str(), cache_size);
}
 
DslReturnType dsl_sink_record_dimensions_get(const wchar_t* name, uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordDimensionsGet(cstrName.c_str(), width, height);
}

DslReturnType dsl_sink_record_dimensions_set(const wchar_t* name, uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordDimensionsSet(cstrName.c_str(), width, height);
}

DslReturnType dsl_sink_record_is_on_get(const wchar_t* name, boolean* is_on)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordIsOnGet(cstrName.c_str(), is_on);
}

DslReturnType dsl_sink_record_reset_done_get(const wchar_t* name, boolean* reset_done)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordResetDoneGet(cstrName.c_str(), reset_done);
}

DslReturnType dsl_sink_record_video_player_add(const wchar_t* name, 
    const wchar_t* player)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(player);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPlayer(player);
    std::string cstrPlayer(wstrPlayer.begin(), wstrPlayer.end());

    return DSL::Services::GetServices()->
        SinkRecordVideoPlayerAdd(cstrName.c_str(), cstrPlayer.c_str());
}
    
DslReturnType dsl_sink_record_video_player_remove(const wchar_t* name, 
    const wchar_t* player)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(player);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPlayer(player);
    std::string cstrPlayer(wstrPlayer.begin(), wstrPlayer.end());

    return DSL::Services::GetServices()->
        SinkRecordVideoPlayerRemove(cstrName.c_str(), cstrPlayer.c_str());
}

DslReturnType dsl_sink_record_mailer_add(const wchar_t* name, 
    const wchar_t* mailer, const wchar_t* subject)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(mailer);
    RETURN_IF_PARAM_IS_NULL(subject);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrMailer(mailer);
    std::string cstrMailer(wstrMailer.begin(), wstrMailer.end());
    std::wstring wstrSubject(subject);
    std::string cstrSubject(wstrSubject.begin(), wstrSubject.end());

    return DSL::Services::GetServices()->SinkRecordMailerAdd(
        cstrName.c_str(), cstrMailer.c_str(), cstrSubject.c_str());
}
    
DslReturnType dsl_sink_record_mailer_remove(const wchar_t* name, 
    const wchar_t* mailer)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(mailer);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrMailer(mailer);
    std::string cstrMailer(wstrMailer.begin(), wstrMailer.end());

    return DSL::Services::GetServices()->SinkRecordMailerRemove(
        cstrName.c_str(), cstrMailer.c_str());
}

DslReturnType dsl_sink_rtmp_new(const wchar_t* name, const wchar_t* uri,
    uint bitrate, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(uri);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUri(uri);
    std::string cstrUri(wstrUri.begin(), wstrUri.end());

    return DSL::Services::GetServices()->SinkRtmpNew(cstrName.c_str(), 
        cstrUri.c_str(), bitrate, interval);
}     

DslReturnType dsl_sink_rtmp_uri_get(const wchar_t* name, const wchar_t** uri)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(uri);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cUri;
    static std::string cstrUri;
    static std::wstring wcstrUri;
    
    uint retval = DSL::Services::GetServices()->SinkRtmpUriGet(cstrName.c_str(), 
        &cUri);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrUri.assign(cUri);
        wcstrUri.assign(cstrUri.begin(), cstrUri.end());
        *uri = wcstrUri.c_str();
    }
    return retval;
}

DslReturnType dsl_sink_rtmp_uri_set(const wchar_t* name, const wchar_t* uri)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(uri);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUri(uri);
    std::string cstrUri(wstrUri.begin(), wstrUri.end());

    return DSL::Services::GetServices()->SinkRtmpUriSet(cstrName.c_str(), 
        cstrUri.c_str());
}

DslReturnType dsl_sink_rtsp_server_new(const wchar_t* name, const wchar_t* host, 
     uint udpPort, uint rtspPort, uint codec, uint bitrate, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(host);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHost(host);
    std::string cstrHost(wstrHost.begin(), wstrHost.end());

    return DSL::Services::GetServices()->SinkRtspServerNew(cstrName.c_str(), 
        cstrHost.c_str(), udpPort, rtspPort, codec, bitrate, interval);
}     

DslReturnType dsl_sink_rtsp_server_settings_get(const wchar_t* name,
    uint* udpPort, uint* rtspPort)
{    
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkRtspServerSettingsGet(cstrName.c_str(), 
        udpPort, rtspPort);
}    

DslReturnType dsl_sink_rtsp_client_new(const wchar_t* name, const wchar_t* uri, 
    uint codec, uint bitrate, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(uri);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUri(uri);
    std::string cstrUri(wstrUri.begin(), wstrUri.end());

    return DSL::Services::GetServices()->SinkRtspClientNew(cstrName.c_str(), 
        cstrUri.c_str(), codec, bitrate, interval);
}     

DslReturnType dsl_sink_rtsp_client_credentials_set(const wchar_t* name, 
    const wchar_t* user_id, const wchar_t* user_pw)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(user_id);
    RETURN_IF_PARAM_IS_NULL(user_pw);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrId(user_id);
    std::string cstrId(wstrId.begin(), wstrId.end());
    std::wstring wstrPassword(user_pw);
    std::string cstrPassword(wstrPassword.begin(), wstrPassword.end());

    return DSL::Services::GetServices()->SinkRtspClientCredentialsSet(
        cstrName.c_str(), cstrId.c_str(), cstrPassword.c_str());
}
    
DslReturnType dsl_sink_rtsp_client_latency_get(const wchar_t* name, 
    uint* latency)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(latency);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRtspClientLatencyGet(
        cstrName.c_str(), latency);
}

DslReturnType dsl_sink_rtsp_client_latency_set(const wchar_t* name, 
    uint latency)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRtspClientLatencySet(
        cstrName.c_str(), latency);
}
     
DslReturnType dsl_sink_rtsp_client_profiles_get(const wchar_t* name,
    uint* profiles)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(profiles);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRtspClientProfilesGet(
        cstrName.c_str(), profiles);
}

DslReturnType dsl_sink_rtsp_client_profiles_set(const wchar_t* name,
    uint profiles)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRtspClientProfilesSet(
        cstrName.c_str(), profiles);
}
     
DslReturnType dsl_sink_rtsp_client_protocols_get(const wchar_t* name,
    uint* protocols)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(protocols);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRtspClientProtocolsGet(
        cstrName.c_str(), protocols);
}

DslReturnType dsl_sink_rtsp_client_protocols_set(const wchar_t* name,
    uint protocols)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRtspClientProtocolsSet(
        cstrName.c_str(), protocols);
}
     
DslReturnType dsl_sink_rtsp_client_tls_validation_flags_get(const wchar_t* name,
    uint* flags)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(flags);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRtspClientTlsValidationFlagsGet(
        cstrName.c_str(), flags);
}

DslReturnType dsl_sink_rtsp_client_tls_validation_flags_set(const wchar_t* name,
    uint flags)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRtspClientTlsValidationFlagsSet(
        cstrName.c_str(), flags);
}
     
DslReturnType dsl_sink_interpipe_new(const wchar_t* name,
    boolean forward_eos, boolean forward_events)
{    
#if !defined(BUILD_INTER_PIPE)
    #error "BUILD_INTER_PIPE must be defined"
#elif BUILD_INTER_PIPE != true
    LOG_ERROR("To use the Inter-Pipe services, set BUILD_INTER_PIPE=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkInterpipeNew(cstrName.c_str(),
        forward_eos, forward_events);
#endif        
}    

DslReturnType dsl_sink_interpipe_forward_settings_get(const wchar_t* name,
    boolean* forward_eos, boolean* forward_events)
{    
#if !defined(BUILD_INTER_PIPE)
    #error "BUILD_INTER_PIPE must be defined"
#elif BUILD_INTER_PIPE != true
    LOG_ERROR("To use the Inter-Pipe services, set BUILD_INTER_PIPE=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(forward_eos);
    RETURN_IF_PARAM_IS_NULL(forward_events);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkInterpipeForwardSettingsGet(
        cstrName.c_str(), forward_eos, forward_events);
#endif        
}    
    
DslReturnType dsl_sink_interpipe_forward_settings_set(const wchar_t* name,
    boolean forward_eos, boolean forward_events)
{    
#if !defined(BUILD_INTER_PIPE)
    #error "BUILD_INTER_PIPE must be defined"
#elif BUILD_INTER_PIPE != true
    LOG_ERROR("To use the Inter-Pipe services, set BUILD_INTER_PIPE=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkInterpipeForwardSettingsSet(
        cstrName.c_str(), forward_eos, forward_events);
#endif        
}    

DslReturnType dsl_sink_interpipe_num_listeners_get(const wchar_t* name,
    uint* num_listeners)
{
#if !defined(BUILD_INTER_PIPE)
    #error "BUILD_INTER_PIPE must be defined"
#elif BUILD_INTER_PIPE != true
    LOG_ERROR("To use the Inter-Pipe services, set BUILD_INTER_PIPE=true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(num_listeners);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkInterpipeNumListenersGet(
        cstrName.c_str(), num_listeners);
#endif    
}    

DslReturnType dsl_sink_image_multi_new(const wchar_t* name, 
    const wchar_t* file_path, uint width, uint height,
    uint fps_n, uint fps_d)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->SinkImageMultiNew(cstrName.c_str(),
        cstrFilePath.c_str(), width, height, fps_n, fps_d);
}

DslReturnType dsl_sink_image_multi_file_path_get(const wchar_t* name, 
    const wchar_t** file_path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cFilePath;
    static std::string cstrFilePath;
    static std::wstring wcstrFilePath;
    
    uint retval = DSL::Services::GetServices()->SinkImageMultiFilePathGet(
        cstrName.c_str(), &cFilePath);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrFilePath.assign(cFilePath);
        wcstrFilePath.assign(cstrFilePath.begin(), cstrFilePath.end());
        *file_path = wcstrFilePath.c_str();
    }
    return retval;
}

DslReturnType dsl_sink_image_multi_file_path_set(const wchar_t* name, 
    const wchar_t* file_path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->SinkImageMultiFilePathSet(cstrName.c_str(), 
        cstrFilePath.c_str());
}

DslReturnType dsl_sink_image_multi_dimensions_get(const wchar_t* name, 
    uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(width);
    RETURN_IF_PARAM_IS_NULL(height);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkImageMultiDimensionsGet(
        cstrName.c_str(), width, height);
}

DslReturnType dsl_sink_image_multi_dimensions_set(const wchar_t* name, 
    uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkImageMultiDimensionsSet(
        cstrName.c_str(), width, height);
}

DslReturnType dsl_sink_image_multi_frame_rate_get(const wchar_t* name, 
    uint* fps_n, uint* fps_d)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(fps_n);
    RETURN_IF_PARAM_IS_NULL(fps_d);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkImageMultiFrameRateGet(
        cstrName.c_str(), fps_n, fps_d);
}
    
DslReturnType dsl_sink_image_multi_frame_rate_set(const wchar_t* name, 
    uint fps_n, uint fps_d)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkImageMultiFrameRateSet(
        cstrName.c_str(), fps_n, fps_d);
}

DslReturnType dsl_sink_image_multi_file_max_get(const wchar_t* name, 
    uint* max)
{    
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(max);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkImageMultiFileMaxGet(
        cstrName.c_str(), max);
}

DslReturnType dsl_sink_image_multi_file_max_set(const wchar_t* name, 
    uint max)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkImageMultiFileMaxSet(
        cstrName.c_str(), max);
}

DslReturnType dsl_sink_frame_capture_new(const wchar_t* name, 
    const wchar_t* frame_capture_action)
{
#if !defined(BUILD_WITH_FFMPEG) || !defined(BUILD_WITH_OPENCV)
    #error "BUILD_WITH_FFMPEG and BUILD_WITH_OPENCV must be defined"
#elif (BUILD_WITH_FFMPEG != true) && (BUILD_WITH_OPENCV != true)
    LOG_ERROR("dsl_sink_frame_capture_new requires one of BUILD_WITH_FFMPEG \
       or BUILD_WITH_OPENCV to be set true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else    
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(frame_capture_action);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAction(frame_capture_action);
    std::string cstrAction(wstrAction.begin(), wstrAction.end());

    return DSL::Services::GetServices()->SinkFrameCaptureNew(
        cstrName.c_str(), cstrAction.c_str());
#endif        
}
    
DslReturnType dsl_sink_frame_capture_initiate(const wchar_t* name)
{
#if !defined(BUILD_WITH_FFMPEG) || !defined(BUILD_WITH_OPENCV)
    #error "BUILD_WITH_FFMPEG and BUILD_WITH_OPENCV must be defined"
#elif (BUILD_WITH_FFMPEG != true) && (BUILD_WITH_OPENCV != true)
    LOG_ERROR("dsl_sink_frame_capture_new requires one of BUILD_WITH_FFMPEG \
       or BUILD_WITH_OPENCV to be set true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else    
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkFrameCaptureInitiate(
        cstrName.c_str());
#endif        
}
    
DslReturnType dsl_sink_frame_capture_schedule(const wchar_t* name,
    uint64_t frame_number)
{
#if !defined(BUILD_WITH_FFMPEG) || !defined(BUILD_WITH_OPENCV)
    #error "BUILD_WITH_FFMPEG and BUILD_WITH_OPENCV must be defined"
#elif (BUILD_WITH_FFMPEG != true) && (BUILD_WITH_OPENCV != true)
    LOG_ERROR("dsl_sink_frame_capture_new requires one of BUILD_WITH_FFMPEG \
       or BUILD_WITH_OPENCV to be set true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else    
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkFrameCaptureSchedule(
        cstrName.c_str(), frame_number);
#endif        
}
 
    
// NOTE: the WebRTC Sink implementation requires DS 1.18.0 or later
DslReturnType dsl_sink_webrtc_new(const wchar_t* name, const wchar_t* stun_server,
    const wchar_t* turn_server, uint codec, uint bitrate, uint interval)
{
#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC != true
    LOG_ERROR("WebRTC & WebSocket services require BUILD_WEBRTC to be set to true \
        in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrStunServer;
    std::string cstrTurnServer;
    if (stun_server != NULL)
    {
        std::wstring wstrStunServer(stun_server);
        cstrStunServer.assign(wstrStunServer.begin(), wstrStunServer.end());
    }
    if (turn_server != NULL)
    {
        std::wstring wstrTurnServer(turn_server);
        cstrTurnServer.assign(wstrTurnServer.begin(), wstrTurnServer.end());
    }

    return DSL::Services::GetServices()->SinkWebRtcNew(cstrName.c_str(),
        cstrStunServer.c_str(), cstrTurnServer.c_str(), codec, bitrate, interval);
#endif    
}

DslReturnType dsl_sink_webrtc_connection_close(const wchar_t* name)
{
#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC != true
    LOG_ERROR("WebRTC & WebSocket services require BUILD_WEBRTC to be set to true \
        in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWebRtcConnectionClose(cstrName.c_str());
#endif    
}

DslReturnType dsl_sink_webrtc_servers_get(const wchar_t* name, 
    const wchar_t** stun_server, const wchar_t** turn_server)
{
#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC != true
    LOG_ERROR("WebRTC & WebSocket services require BUILD_WEBRTC to be set to true \
        in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cStunServer;
    static std::string cstrStunServer;
    static std::wstring wcstrStunServer;
    const char* cTurnServer;
    static std::string cstrTurnServer;
    static std::wstring wcstrTurnServer;
    
    uint retval = DSL::Services::GetServices()->SinkWebRtcServersGet(cstrName.c_str(), 
        &cStunServer, &cTurnServer);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrStunServer.assign(cStunServer);
        wcstrStunServer.assign(cstrStunServer.begin(), cstrStunServer.end());
        *stun_server = wcstrStunServer.c_str();
        cstrTurnServer.assign(cTurnServer);
        wcstrTurnServer.assign(cstrTurnServer.begin(), cstrTurnServer.end());
        *turn_server = wcstrTurnServer.c_str();
    }
    return retval;

#endif    
}

DslReturnType dsl_sink_webrtc_servers_set(const wchar_t* name, 
    const wchar_t* stun_server, const wchar_t* turn_server)
{
#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC != true
    LOG_ERROR("WebRTC & WebSocket services require BUILD_WEBRTC to be set to true \
        in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrStunServer;
    std::string cstrTurnServer;
    if (stun_server != NULL)
    {
        std::wstring wstrStunServer(stun_server);
        cstrStunServer.assign(wstrStunServer.begin(), wstrStunServer.end());
    }
    if (turn_server != NULL)
    {
        std::wstring wstrTurnServer(turn_server);
        cstrTurnServer.assign(wstrTurnServer.begin(), wstrTurnServer.end());
    }

    return DSL::Services::GetServices()->SinkWebRtcServersSet(cstrName.c_str(),
        cstrStunServer.c_str(), cstrTurnServer.c_str());
#endif    
}

DslReturnType dsl_sink_webrtc_client_listener_add(const wchar_t* name, 
    dsl_sink_webrtc_client_listener_cb listener, void* client_data)
{
#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC != true
    LOG_ERROR("WebRTC & WebSocket services require BUILD_WEBRTC to be set to true \
        in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        SinkWebRtcClientListenerAdd(cstrName.c_str(), listener, client_data);
#endif    
}

DslReturnType dsl_sink_webrtc_client_listener_remove(const wchar_t* name, 
    dsl_sink_webrtc_client_listener_cb listener)
{
#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC != true
    LOG_ERROR("WebRTC & WebSocket services require BUILD_WEBRTC to be set to true \
        in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        SinkWebRtcClientListenerRemove(cstrName.c_str(), listener);
#endif    
}

DslReturnType dsl_websocket_server_path_add(const wchar_t* path)
{
#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC != true
    LOG_ERROR("WebRTC & WebSocket services require BUILD_WEBRTC to be set to true \
        in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(path);

    std::wstring wstrPath(path);
    std::string cstrPath(wstrPath.begin(), wstrPath.end());

    return DSL::Services::GetServices()->
        WebsocketServerPathAdd(cstrPath.c_str());
#endif    
}

DslReturnType dsl_websocket_server_listening_start(uint port_number)
{
#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC != true
    LOG_ERROR("WebRTC & WebSocket services require BUILD_WEBRTC to be set to true \
        in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    return DSL::Services::GetServices()->
        WebsocketServerListeningStart(port_number);
#endif    
}

DslReturnType dsl_websocket_server_listening_stop()
{
#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC != true
    LOG_ERROR("WebRTC & WebSocket services require BUILD_WEBRTC to be set to true \
        in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    return DSL::Services::GetServices()->
        WebsocketServerListeningStop();
#endif    
}

DslReturnType dsl_websocket_server_listening_state_get(boolean* is_listening,
    uint* port_number)
{
#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC != true
    LOG_ERROR("WebRTC & WebSocket services require BUILD_WEBRTC to be set to true \
        in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    return DSL::Services::GetServices()->
        WebsocketServerListeningStateGet(is_listening, port_number);
#endif    
}

DslReturnType dsl_websocket_server_client_listener_add( 
    dsl_websocket_server_client_listener_cb listener, void* client_data)
{
#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC != true
    LOG_ERROR("WebRTC & WebSocket services require BUILD_WEBRTC to be set to true \
        in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(listener);

    return DSL::Services::GetServices()->
        WebsocketServerClientListenerAdd(listener, client_data);
#endif    
}

DslReturnType dsl_websocket_server_client_listener_remove( 
    dsl_websocket_server_client_listener_cb listener)
{
#if !defined(BUILD_WEBRTC)
    #error "BUILD_WEBRTC must be defined"
#elif BUILD_WEBRTC != true
    LOG_ERROR("WebRTC & WebSocket services require BUILD_WEBRTC to be set to true \
        in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else
    RETURN_IF_PARAM_IS_NULL(listener);

    return DSL::Services::GetServices()->
        WebsocketServerClientListenerRemove(listener);
#endif    
}

DslReturnType dsl_sink_message_new(const wchar_t* name, 
    const wchar_t* converter_config_file, uint payload_type, 
    const wchar_t* broker_config_file, const wchar_t* protocol_lib, 
    const wchar_t* connection_string, const wchar_t* topic)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(broker_config_file);
    RETURN_IF_PARAM_IS_NULL(protocol_lib);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrConvConfig;
    if (converter_config_file)
    {
        std::wstring wstrConvConfig(converter_config_file);
        cstrConvConfig.assign(wstrConvConfig.begin(), wstrConvConfig.end());
    }
    std::wstring wstrBrokerConfig(broker_config_file);
    std::string cstrBrokerConfig(wstrBrokerConfig.begin(), wstrBrokerConfig.end());
    std::wstring wstrProtocolLib(protocol_lib);
    std::string cstrProtocolLib(wstrProtocolLib.begin(), wstrProtocolLib.end());
    
    std::string cstrConn;
    if (connection_string != NULL)
    {
        std::wstring wstrConn(connection_string);
        cstrConn.assign(wstrConn.begin(), wstrConn.end());
    }
    
    std::string cstrTopic;
    if (topic != NULL)
    {
        std::wstring wstrTopic(topic);
        cstrTopic.assign(wstrTopic.begin(), wstrTopic.end());
    }

    return DSL::Services::GetServices()->SinkMessageNew(cstrName.c_str(), 
        cstrConvConfig.c_str(), payload_type, cstrBrokerConfig.c_str(), 
        cstrProtocolLib.c_str(), cstrConn.c_str(), cstrTopic.c_str());
}

DslReturnType dsl_sink_message_meta_type_get(const wchar_t* name,
    uint* meta_type)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkMessageMetaTypeGet(cstrName.c_str(), 
        meta_type);
}
    
DslReturnType dsl_sink_message_meta_type_set(const wchar_t* name,
    uint meta_type)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkMessageMetaTypeSet(cstrName.c_str(), 
        meta_type);
}
    
DslReturnType dsl_sink_message_converter_settings_get(const wchar_t* name, 
    const wchar_t** converter_config_file, uint* payload_type)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(converter_config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cConfigFile;
    static std::string cstrConfigFile;
    static std::wstring wcstrConfigFile;
    
    uint retval = DSL::Services::GetServices()->SinkMessageConverterSettingsGet(
        cstrName.c_str(), &cConfigFile, payload_type);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrConfigFile.assign(cConfigFile);
        wcstrConfigFile.assign(cstrConfigFile.begin(), cstrConfigFile.end());
        *converter_config_file = wcstrConfigFile.c_str();
    }
    return retval;
}
    
DslReturnType dsl_sink_message_converter_settings_set(const wchar_t* name, 
    const wchar_t* converter_config_file, uint payload_type)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(converter_config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConvConfig(converter_config_file);
    std::string cstrConvConfig(wstrConvConfig.begin(), wstrConvConfig.end());

    return DSL::Services::GetServices()->SinkMessageConverterSettingsSet(cstrName.c_str(), 
        cstrConvConfig.c_str(), payload_type);
}

DslReturnType dsl_sink_message_broker_settings_get(const wchar_t* name, 
    const wchar_t** broker_config_file, const wchar_t**  protocol_lib,
    const wchar_t** connection_string, const wchar_t** topic)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(broker_config_file);
    RETURN_IF_PARAM_IS_NULL(protocol_lib);
    RETURN_IF_PARAM_IS_NULL(connection_string);
    RETURN_IF_PARAM_IS_NULL(topic);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cConfigFile;
    static std::string cstrConfigFile;
    static std::wstring wcstrConfigFile;
    const char* cProtocolLib;
    static std::string cstrProtocolLib;
    static std::wstring wcstrProtocolLib;
    const char* cConnStr;
    static std::string cstrConnStr;
    static std::wstring wcstrConnStr;
    const char* cTopic;
    static std::string cstrTopic;
    static std::wstring wcstrTopic;
    
    uint retval = DSL::Services::GetServices()->SinkMessageBrokerSettingsGet(
        cstrName.c_str(), &cConfigFile, &cProtocolLib, &cConnStr, &cTopic);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrConfigFile.assign(cConfigFile);
        wcstrConfigFile.assign(cstrConfigFile.begin(), cstrConfigFile.end());
        *broker_config_file = wcstrConfigFile.c_str();
        cstrProtocolLib.assign(cProtocolLib);
        wcstrProtocolLib.assign(cstrProtocolLib.begin(), cstrProtocolLib.end());
        *protocol_lib = wcstrProtocolLib.c_str();
        cstrConnStr.assign(cConnStr);
        wcstrConnStr.assign(cstrConnStr.begin(), cstrConnStr.end());
        *connection_string = wcstrConnStr.c_str();
        cstrTopic.assign(cTopic);
        wcstrTopic.assign(cstrTopic.begin(), cstrTopic.end());
        *topic = wcstrTopic.c_str();
    }
    return retval;
} 

DslReturnType dsl_sink_message_broker_settings_set(const wchar_t* name, 
    const wchar_t* broker_config_file, const wchar_t* protocol_lib,
    const wchar_t* connection_string, const wchar_t* topic)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(broker_config_file);
    RETURN_IF_PARAM_IS_NULL(protocol_lib);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrBrokerConfig(broker_config_file);
    std::string cstrBrokerConfig(wstrBrokerConfig.begin(), wstrBrokerConfig.end());
    std::wstring wstrProtocolLib(protocol_lib);
    std::string cstrProtocolLib(wstrProtocolLib.begin(), wstrProtocolLib.end());
    

    std::string cstrConn;
    if (connection_string != NULL)
    {
        std::wstring wstrConn(connection_string);
        cstrConn.assign(wstrConn.begin(), wstrConn.end());
    }
    
    std::string cstrTopic;
    if (topic != NULL)
    {
        std::wstring wstrTopic(topic);
        cstrTopic.assign(wstrTopic.begin(), wstrTopic.end());
    }

    return DSL::Services::GetServices()->SinkMessageBrokerSettingsSet(cstrName.c_str(), 
        cstrBrokerConfig.c_str(), cstrProtocolLib.c_str(), 
        cstrConn.c_str(), cstrTopic.c_str());
}

DslReturnType dsl_sink_message_payload_debug_dir_get(const wchar_t* name, 
    const wchar_t** debug_dir)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(debug_dir);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cDebugDir;
    static std::string cstrDebugDir;
    static std::wstring wcstrDebugDir;
    
    uint retval = DSL::Services::GetServices()->GetSinkMessagePayloadDebugDirGet(
        cstrName.c_str(), &cDebugDir);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrDebugDir.assign(cDebugDir);
        wcstrDebugDir.assign(cstrDebugDir.begin(), cstrDebugDir.end());
        *debug_dir = wcstrDebugDir.c_str();
    }
    return retval;
}

DslReturnType dsl_sink_message_payload_debug_dir_set(const wchar_t* name, 
    const wchar_t* debug_dir)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(debug_dir);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrDebugDir(debug_dir);
    std::string cstrDebugDir(wstrDebugDir.begin(), wstrDebugDir.end());

    return DSL::Services::GetServices()->GetSinkMessagePayloadDebugDirSet(
        cstrName.c_str(), cstrDebugDir.c_str());
}
    
DslReturnType dsl_sink_webrtc_livekit_new(const wchar_t* name, 
    const wchar_t* url, const wchar_t* api_key, const wchar_t* secret_key, 
    const wchar_t* room, const wchar_t* identity, const wchar_t* participant)
{
#if !defined(BUILD_LIVEKIT_WEBRTC)
    #error "BUILD_LIVEKIT_WEBRTC must be defined"
#elif (BUILD_LIVEKIT_WEBRTC != true)
    LOG_ERROR("dsl_sink_webrtc_livekit_new requires BUILD_LIVEKIT_WEBRTC \
       to be set true in the Makefile");
    return DSL_RESULT_API_NOT_SUPPORTED;
#else    
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(url);
    RETURN_IF_PARAM_IS_NULL(api_key);
    RETURN_IF_PARAM_IS_NULL(secret_key);
    RETURN_IF_PARAM_IS_NULL(room);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUrl(url);
    std::string cstrUrl(wstrUrl.begin(), wstrUrl.end());
    std::wstring wstrApiKey(api_key);
    std::string cstrApiKey(wstrApiKey.begin(), wstrApiKey.end());
    std::wstring wstrSecretKey(secret_key);
    std::string cstrSecretKey(wstrSecretKey.begin(), wstrSecretKey.end());
    std::wstring wstrRoom(room);
    std::string cstrRoom(wstrRoom.begin(), wstrRoom.end());
    

    std::string cstrIdentity;
    if (identity != NULL)
    {
        std::wstring wstrIdentity(identity);
        cstrIdentity.assign(wstrIdentity.begin(), wstrIdentity.end());
    }
    
    std::string cstrParticipant;
    if (participant != NULL)
    {
        std::wstring wstrParticipant(participant);
        cstrParticipant.assign(wstrParticipant.begin(), wstrParticipant.end());
    }

    return DSL::Services::GetServices()->SinkWebRtcLiveKitNew(cstrName.c_str(), 
        cstrUrl.c_str(), cstrApiKey.c_str(), cstrSecretKey.c_str(),
        cstrRoom.c_str(), cstrIdentity.c_str(), cstrParticipant.c_str());
#endif        
}
    
DslReturnType dsl_sink_v4l2_new(const wchar_t* name, 
    const wchar_t* device_location)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(device_location);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrDevice(device_location);
    std::string cstrDevice(wstrDevice.begin(), wstrDevice.end());

    return DSL::Services::GetServices()->SinkV4l2New(cstrName.c_str(), 
        cstrDevice.c_str());
}     

DslReturnType dsl_sink_v4l2_device_location_get(const wchar_t* name,
    const wchar_t** device_location)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(device_location);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cDeviceLocation;
    static std::string cstrDeviceLocation;
    static std::wstring wcstrDeviceLocation;
    
    uint retval = DSL::Services::GetServices()->SinkV4l2DeviceLocationGet(
        cstrName.c_str(), &cDeviceLocation);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrDeviceLocation.assign(cDeviceLocation);
        wcstrDeviceLocation.assign(cstrDeviceLocation.begin(), 
            cstrDeviceLocation.end());
        *device_location = wcstrDeviceLocation.c_str();
    }
    return retval;
}

DslReturnType dsl_sink_v4l2_device_location_set(const wchar_t* name,
    const wchar_t* device_location)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(device_location);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrDeviceLocation(device_location);
    std::string cstrDeviceLocation(wstrDeviceLocation.begin(), 
        wstrDeviceLocation.end());
    
    return DSL::Services::GetServices()->SinkV4l2DeviceLocationSet(
        cstrName.c_str(), cstrDeviceLocation.c_str());
}
   
DslReturnType dsl_sink_v4l2_device_name_get(const wchar_t* name,
    const wchar_t** device_name)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(device_name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cDeviceName;
    static std::string cstrDeviceName;
    static std::wstring wcstrDeviceName;
    
    uint retval = DSL::Services::GetServices()->SinkV4l2DeviceNameGet(
        cstrName.c_str(), &cDeviceName);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrDeviceName.assign(cDeviceName);
        wcstrDeviceName.assign(cstrDeviceName.begin(), cstrDeviceName.end());
        *device_name = wcstrDeviceName.c_str();
    }
    return retval;
}
    
DslReturnType dsl_sink_v4l2_device_fd_get(const wchar_t* name,
    int* device_fd)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(device_fd);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkV4l2DeviceFdGet(
        cstrName.c_str(), device_fd);
}

DslReturnType dsl_sink_v4l2_device_flags_get(const wchar_t* name,
    uint* device_flags)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(device_flags);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkV4l2DeviceFlagsGet(
        cstrName.c_str(), device_flags);
}
    
DslReturnType dsl_sink_v4l2_buffer_in_format_get(const wchar_t* name,
    const wchar_t** format)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(format);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cFormat;
    static std::string cstrFormat;
    static std::wstring wcstrFormat;
    
    uint retval = DSL::Services::GetServices()->SinkV4l2BufferInFormatGet(
        cstrName.c_str(), &cFormat);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrFormat.assign(cFormat);
        wcstrFormat.assign(cstrFormat.begin(), 
            cstrFormat.end());
        *format = wcstrFormat.c_str();
    }
    return retval;
}


DslReturnType dsl_sink_v4l2_buffer_in_format_set(const wchar_t* name,
    const wchar_t* format)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(format);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFormat(format);
    std::string cstrFormat(wstrFormat.begin(), 
        wstrFormat.end());
    
    return DSL::Services::GetServices()->SinkV4l2BufferInFormatSet(
        cstrName.c_str(), cstrFormat.c_str());
}

DslReturnType dsl_sink_v4l2_picture_settings_get(const wchar_t* name,
    int* brightness, int* contrast, int* saturation)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(brightness);
    RETURN_IF_PARAM_IS_NULL(contrast);
    RETURN_IF_PARAM_IS_NULL(saturation);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkV4l2PictureSettingsGet(
        cstrName.c_str(), brightness, contrast, saturation);
}

DslReturnType dsl_sink_v4l2_picture_settings_set(const wchar_t* name,
    int brightness, int contrast, int saturation)    
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkV4l2PictureSettingsSet(
        cstrName.c_str(), brightness, contrast, saturation);
}
    
DslReturnType dsl_sink_sync_enabled_get(const wchar_t* name, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkSyncEnabledGet(cstrName.c_str(), 
        enabled);
}
    
DslReturnType dsl_sink_sync_enabled_set(const wchar_t* name, boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkSyncEnabledSet(cstrName.c_str(), 
        enabled);
}
    
DslReturnType dsl_sink_async_enabled_get(const wchar_t* name, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkAsyncEnabledGet(cstrName.c_str(), 
        enabled);
}
    
DslReturnType dsl_sink_async_enabled_set(const wchar_t* name, boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkAsyncEnabledSet(cstrName.c_str(), 
        enabled);
}
    
DslReturnType dsl_sink_max_lateness_get(const wchar_t* name, int64_t* max_lateness)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(max_lateness);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkMaxLatenessGet(cstrName.c_str(), 
        max_lateness);
}
    
DslReturnType dsl_sink_max_lateness_set(const wchar_t* name, int64_t max_lateness)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkMaxLatenessSet(cstrName.c_str(), 
        max_lateness);
}
    
DslReturnType dsl_sink_qos_enabled_get(const wchar_t* name, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkQosEnabledGet(cstrName.c_str(), 
        enabled);
}
    
DslReturnType dsl_sink_qos_enabled_set(const wchar_t* name, boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkQosEnabledSet(cstrName.c_str(), 
        enabled);
}
    
DslReturnType dsl_sink_pph_add(const wchar_t* name, const wchar_t* handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->SinkPphAdd(cstrName.c_str(), 
        cstrHandler.c_str());
}

DslReturnType dsl_sink_pph_remove(const wchar_t* name,
    const wchar_t* handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->SinkPphRemove(cstrName.c_str(), 
        cstrHandler.c_str());
}

DslReturnType dsl_component_delete(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->ComponentDelete(cstrName.c_str());
}

DslReturnType dsl_component_delete_many(const wchar_t** names)
{
    RETURN_IF_PARAM_IS_NULL(names);

    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());
        DslReturnType retval = DSL::Services::GetServices()->ComponentDelete(cstrName.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_component_delete_all()
{
    return DSL::Services::GetServices()->ComponentDeleteAll();
}

uint dsl_component_list_size()
{
    return DSL::Services::GetServices()->ComponentListSize();
}

DslReturnType dsl_component_gpuid_get(const wchar_t* name, uint* gpuid)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->ComponentGpuIdGet(cstrName.c_str(), gpuid);
}

DslReturnType dsl_component_gpuid_set(const wchar_t* name, uint gpuid)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->ComponentGpuIdSet(cstrName.c_str(), gpuid);
}

DslReturnType dsl_component_gpuid_set_many(const wchar_t** names, uint gpuid)
{
    RETURN_IF_PARAM_IS_NULL(names);

    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());
        DslReturnType retval = DSL::Services::GetServices()->ComponentGpuIdSet(
                cstrName.c_str(), gpuid);
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_component_nvbuf_mem_type_get(const wchar_t* name, uint* type)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->ComponentNvbufMemTypeGet(cstrName.c_str(), type);
}

DslReturnType dsl_component_nvbuf_mem_type_set(const wchar_t* name, uint type)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->ComponentNvbufMemTypeSet(cstrName.c_str(), type);
}

DslReturnType dsl_component_nvbuf_mem_type_set_many(const wchar_t** names, uint type)
{
    RETURN_IF_PARAM_IS_NULL(names);

    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());
        DslReturnType retval = DSL::Services::GetServices()->ComponentNvbufMemTypeSet(
            cstrName.c_str(), type);
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_branch_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->BranchNew(cstrName.c_str());
}

DslReturnType dsl_branch_new_component_add_many(const wchar_t* name, 
    const wchar_t** components)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(components);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    DslReturnType retval = DSL::Services::GetServices()->BranchNew(cstrName.c_str());
    if (retval != DSL_RESULT_SUCCESS)
    {
        return retval;
    }

    for (const wchar_t** component = components; *component; component++)
    {
        std::wstring wstrComponent(*component);
        std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());
        retval = DSL::Services::GetServices()->BranchComponentAdd(cstrName.c_str(), cstrComponent.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_branch_new_many(const wchar_t** names)
{
    RETURN_IF_PARAM_IS_NULL(names);

    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());
        DslReturnType retval = DSL::Services::GetServices()->BranchNew(cstrName.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}


DslReturnType dsl_branch_component_add(const wchar_t* name, 
    const wchar_t* component)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(component);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrComponent(component);
    std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());

    return DSL::Services::GetServices()->BranchComponentAdd(cstrName.c_str(), cstrComponent.c_str());
}

DslReturnType dsl_branch_component_add_many(const wchar_t* name, 
    const wchar_t** components)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(components);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    for (const wchar_t** component = components; *component; component++)
    {
        std::wstring wstrComponent(*component);
        std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());
        DslReturnType retval = DSL::Services::GetServices()->BranchComponentAdd(cstrName.c_str(), cstrComponent.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_branch_component_remove(const wchar_t* name, 
    const wchar_t* component)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(component);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrComponent(component);
    std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());

    return DSL::Services::GetServices()->BranchComponentRemove(cstrName.c_str(), cstrComponent.c_str());
}

DslReturnType dsl_branch_component_remove_many(const wchar_t* name, 
    const wchar_t** components)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(components);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    for (const wchar_t** component = components; *component; component++)
    {
        std::wstring wstrComponent(*component);
        std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());
        DslReturnType retval = DSL::Services::GetServices()->BranchComponentRemove(cstrName.c_str(), cstrComponent.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_pipeline_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineNew(cstrName.c_str());
}

DslReturnType dsl_pipeline_new_component_add_many(const wchar_t* name, 
    const wchar_t** components)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(components);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    DslReturnType retval = DSL::Services::GetServices()->PipelineNew(cstrName.c_str());
    if (retval != DSL_RESULT_SUCCESS)
    {
        return retval;
    }
    for (const wchar_t** component = components; *component; component++)
    {
        std::wstring wstrComponent(*component);
        std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());
        DslReturnType retval = DSL::Services::GetServices()->PipelineComponentAdd(cstrName.c_str(), cstrComponent.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_pipeline_new_many(const wchar_t** names)
{
    RETURN_IF_PARAM_IS_NULL(names);

    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());
        DslReturnType retval = DSL::Services::GetServices()->PipelineNew(cstrName.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_pipeline_delete(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineDelete(cstrName.c_str());
}

DslReturnType dsl_pipeline_delete_many(const wchar_t** names)
{
    RETURN_IF_PARAM_IS_NULL(names);

    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());
        DslReturnType retval = DSL::Services::GetServices()->PipelineDelete(cstrName.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_pipeline_delete_all()
{
    return DSL::Services::GetServices()->PipelineDeleteAll();
}

uint dsl_pipeline_list_size()
{
    return DSL::Services::GetServices()->PipelineListSize();
}

DslReturnType dsl_pipeline_component_add(const wchar_t* name, 
    const wchar_t* component)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(component);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrComponent(component);
    std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());

    return DSL::Services::GetServices()->PipelineComponentAdd(cstrName.c_str(), cstrComponent.c_str());
}

DslReturnType dsl_pipeline_component_add_many(const wchar_t* name, 
    const wchar_t** components)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(components);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    for (const wchar_t** component = components; *component; component++)
    {
        std::wstring wstrComponent(*component);
        std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());
        DslReturnType retval = DSL::Services::GetServices()->PipelineComponentAdd(cstrName.c_str(), cstrComponent.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_pipeline_component_remove(const wchar_t* name, 
    const wchar_t* component)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(component);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrComponent(component);
    std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());

    return DSL::Services::GetServices()->PipelineComponentRemove(cstrName.c_str(), cstrComponent.c_str());
}

DslReturnType dsl_pipeline_component_remove_many(const wchar_t* name, 
    const wchar_t** components)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(components);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    for (const wchar_t** component = components; *component; component++)
    {
        std::wstring wstrComponent(*component);
        std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());
        DslReturnType retval = 
            DSL::Services::GetServices()->PipelineComponentRemove(cstrName.c_str(), 
                cstrComponent.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

//------------------------------------------------------------------------------------
// NEW NVSTREAMMUX SERVICES - Start
//------------------------------------------------------------------------------------

DslReturnType dsl_pipeline_streammux_config_file_get(const wchar_t* name, 
    const wchar_t** config_file)
{
    RETURN_IF_NEW_NVSTREAMMUX_DISABLED();
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cConfig;
    static std::string cstrConfig;
    static std::wstring wcstrConfig;
    
    uint retval = DSL::Services::GetServices()->PipelineStreammuxConfigFileGet(
        cstrName.c_str(), &cConfig);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrConfig.assign(cConfig);
        wcstrConfig.assign(cstrConfig.begin(), cstrConfig.end());
        *config_file = wcstrConfig.c_str();
    }
    return retval;
}

DslReturnType dsl_pipeline_streammux_config_file_set(const wchar_t* name, 
    const wchar_t* config_file)
{
    RETURN_IF_NEW_NVSTREAMMUX_DISABLED();
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(config_file);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());

    return DSL::Services::GetServices()->PipelineStreammuxConfigFileSet(
        cstrName.c_str(), cstrConfig.c_str());
}

DslReturnType dsl_pipeline_streammux_batch_size_get(const wchar_t* name, 
    uint* batch_size)
{
    RETURN_IF_NEW_NVSTREAMMUX_DISABLED();
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(batch_size);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxBatchSizeGet(
        cstrName.c_str(), batch_size);
}

DslReturnType dsl_pipeline_streammux_batch_size_set(const wchar_t* name, 
    uint batch_size)
{
    RETURN_IF_NEW_NVSTREAMMUX_DISABLED();
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxBatchSizeSet(
        cstrName.c_str(), batch_size);
}

//------------------------------------------------------------------------------------
// NEW NVSTREAMMUX SERVICES - End
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
// OLD NVSTREAMMUX SERVICES - Start
//------------------------------------------------------------------------------------
DslReturnType dsl_pipeline_streammux_batch_properties_get(const wchar_t* name, 
    uint* batch_size, int* batch_timeout)
{
    RETURN_IF_NEW_NVSTREAMMUX_ENABLED();
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(batch_size);
    RETURN_IF_PARAM_IS_NULL(batch_timeout);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxBatchPropertiesGet(
        cstrName.c_str(), batch_size, batch_timeout);
}

DslReturnType dsl_pipeline_streammux_batch_properties_set(const wchar_t* name, 
    uint batch_size, int batch_timeout)
{
    RETURN_IF_NEW_NVSTREAMMUX_ENABLED();
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxBatchPropertiesSet(
        cstrName.c_str(), batch_size, batch_timeout);
}

DslReturnType dsl_pipeline_streammux_dimensions_get(const wchar_t* name, 
    uint* width, uint* height)
{
    RETURN_IF_NEW_NVSTREAMMUX_ENABLED();
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(width);
    RETURN_IF_PARAM_IS_NULL(height);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxDimensionsGet(
        cstrName.c_str(), width, height);
}

DslReturnType dsl_pipeline_streammux_dimensions_set(const wchar_t* name, 
    uint width, uint height)
{
    RETURN_IF_NEW_NVSTREAMMUX_ENABLED();
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxDimensionsSet(
        cstrName.c_str(), width, height);
}    

DslReturnType dsl_pipeline_streammux_padding_get(const wchar_t* name, 
    boolean* enabled)
{
    RETURN_IF_NEW_NVSTREAMMUX_ENABLED();
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxPaddingGet(
        cstrName.c_str(), enabled);
}

DslReturnType dsl_pipeline_streammux_padding_set(const wchar_t* name, 
    boolean enabled)
{
    RETURN_IF_NEW_NVSTREAMMUX_ENABLED();
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxPaddingSet(
        cstrName.c_str(), enabled);
}

DslReturnType dsl_pipeline_streammux_nvbuf_mem_type_get(const wchar_t* name, 
    uint* type)
{
    RETURN_IF_NEW_NVSTREAMMUX_ENABLED();
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(type);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxNvbufMemTypeGet(cstrName.c_str(),
        type);
}

DslReturnType dsl_pipeline_streammux_nvbuf_mem_type_set(const wchar_t* name, 
    uint type)
{
    RETURN_IF_NEW_NVSTREAMMUX_ENABLED();
    RETURN_IF_NEW_NVSTREAMMUX_ENABLED();
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxNvbufMemTypeSet(cstrName.c_str(),
        type);
}

DslReturnType dsl_pipeline_streammux_gpuid_get(const wchar_t* name, uint* gpuid)
{
    RETURN_IF_NEW_NVSTREAMMUX_ENABLED();
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(gpuid);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxGpuIdGet(
        cstrName.c_str(), gpuid);
}

DslReturnType dsl_pipeline_streammux_gpuid_set(const wchar_t* name, uint gpuid)
{
    RETURN_IF_NEW_NVSTREAMMUX_ENABLED();
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxGpuIdSet(
        cstrName.c_str(), gpuid);
}


//------------------------------------------------------------------------------------
// OLD NVSTREAMMUX SERVICES - End
//------------------------------------------------------------------------------------

DslReturnType dsl_pipeline_streammux_num_surfaces_per_frame_get(
    const wchar_t* name, uint* num)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(num);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxNumSurfacesPerFrameGet(
        cstrName.c_str(), num);
}

DslReturnType dsl_pipeline_streammux_num_surfaces_per_frame_set(
    const wchar_t* name, uint num)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxNumSurfacesPerFrameSet(
        cstrName.c_str(), num);
}

DslReturnType dsl_pipeline_streammux_attach_sys_ts_enabled_get(const wchar_t* name, 
    boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxAttachSysTsEnabledGet(
        cstrName.c_str(), enabled);
}

DslReturnType dsl_pipeline_streammux_attach_sys_ts_enabled_set(const wchar_t* name, 
    boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxAttachSysTsEnabledSet(
        cstrName.c_str(), enabled);
}

DslReturnType dsl_pipeline_streammux_sync_inputs_enabled_get(const wchar_t* name, 
    boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxSyncInputsEnabledGet(
        cstrName.c_str(), enabled);
}

DslReturnType dsl_pipeline_streammux_sync_inputs_enabled_set(const wchar_t* name, 
    boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxSyncInputsEnabledSet(
        cstrName.c_str(), enabled);
}

DslReturnType dsl_pipeline_streammux_max_latency_get(const wchar_t* name, 
    uint* max_latency)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(max_latency);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxMaxLatencyGet(
        cstrName.c_str(), max_latency);
}

DslReturnType dsl_pipeline_streammux_max_latency_set(const wchar_t* name, 
    uint max_latency)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxMaxLatencySet(
        cstrName.c_str(), max_latency);
}

DslReturnType dsl_pipeline_streammux_tiler_add(const wchar_t* name, 
    const wchar_t* tiler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(tiler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrTiler(tiler);
    std::string cstrTiler(wstrTiler.begin(), wstrTiler.end());

    return DSL::Services::GetServices()->PipelineStreammuxTilerAdd(
        cstrName.c_str(), cstrTiler.c_str());
}
    
DslReturnType dsl_pipeline_streammux_tiler_remove(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStreammuxTilerRemove(
        cstrName.c_str());
}

DslReturnType dsl_pipeline_streammux_pph_add(const wchar_t* name, 
    const wchar_t* handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->PipelineStreammuxPphAdd(cstrName.c_str(), 
        cstrHandler.c_str());
}

DslReturnType dsl_pipeline_streammux_pph_remove(const wchar_t* name, 
    const wchar_t* handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->PipelineStreammuxPphRemove(cstrName.c_str(), 
        cstrHandler.c_str());
}
 
DslReturnType dsl_pipeline_link_method_get(const wchar_t* name, uint* link_method)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(link_method);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineLinkMethodGet(cstrName.c_str(),
        link_method);
}

DslReturnType dsl_pipeline_link_method_set(const wchar_t* name, uint link_method)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineLinkMethodSet(cstrName.c_str(),
        link_method);
}

DslReturnType dsl_pipeline_pause(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelinePause(cstrName.c_str());
}

DslReturnType dsl_pipeline_play(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelinePlay(cstrName.c_str());
}

DslReturnType dsl_pipeline_stop(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStop(cstrName.c_str());
}

DslReturnType dsl_pipeline_state_get(const wchar_t* name, uint* state)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(state);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineStateGet(cstrName.c_str(), state);
}

DslReturnType dsl_pipeline_is_live(const wchar_t* name, boolean* is_live)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(is_live);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PipelineIsLive(cstrName.c_str(), is_live);
}

DslReturnType dsl_pipeline_dump_to_dot(const wchar_t* name, 
    const wchar_t* filename)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(filename);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilename(filename);
    std::string cstrFilename(wstrFilename.begin(), wstrFilename.end());

    return DSL::Services::GetServices()->PipelineDumpToDot(cstrName.c_str(), 
        cstrFilename.c_str());
}

DslReturnType dsl_pipeline_dump_to_dot_with_ts(const wchar_t* name, 
    const wchar_t* filename)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(filename);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilename(filename);
    std::string cstrFilename(wstrFilename.begin(), wstrFilename.end());

    return DSL::Services::GetServices()->PipelineDumpToDotWithTs(cstrName.c_str(), 
        cstrFilename.c_str());
}

DslReturnType dsl_pipeline_state_change_listener_add(const wchar_t* name, 
    dsl_state_change_listener_cb listener, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        PipelineStateChangeListenerAdd(cstrName.c_str(), listener, client_data);
}

DslReturnType dsl_pipeline_state_change_listener_remove(const wchar_t* name, 
    dsl_state_change_listener_cb listener)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        PipelineStateChangeListenerRemove(cstrName.c_str(), listener);
}

DslReturnType dsl_pipeline_eos_listener_add(const wchar_t* name, 
    dsl_eos_listener_cb listener, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        PipelineEosListenerAdd(cstrName.c_str(), listener, client_data);
}

DslReturnType dsl_pipeline_eos_listener_remove(const wchar_t* name, 
    dsl_eos_listener_cb listener)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        PipelineEosListenerRemove(cstrName.c_str(), listener);
}

DslReturnType dsl_pipeline_error_message_handler_add(const wchar_t* name, 
    dsl_error_message_handler_cb handler, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        PipelineErrorMessageHandlerAdd(cstrName.c_str(), handler, client_data);
}

DslReturnType dsl_pipeline_error_message_handler_remove(const wchar_t* name, 
    dsl_error_message_handler_cb handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        PipelineErrorMessageHandlerRemove(cstrName.c_str(), handler);
}

DslReturnType dsl_pipeline_error_message_last_get(const wchar_t* name, 
    const wchar_t** source, const wchar_t** message)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(source);
    RETURN_IF_PARAM_IS_NULL(message);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    static std::wstring wstrSource;
    static std::wstring wstrMessage;
    
    uint retval = DSL::Services::GetServices()->PipelineErrorMessageLastGet(
        cstrName.c_str(), wstrSource, wstrMessage);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        
        *source = (wstrSource.size()) ? wstrSource.c_str() : NULL;
        *message = (wstrMessage.size()) ? wstrMessage.c_str() : NULL;
    }
    return retval;
}
    
DslReturnType dsl_pipeline_main_loop_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        PipelineMainLoopNew(cstrName.c_str());
}

DslReturnType dsl_pipeline_main_loop_run(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        PipelineMainLoopRun(cstrName.c_str());
}

DslReturnType dsl_pipeline_main_loop_quit(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        PipelineMainLoopQuit(cstrName.c_str());
}

DslReturnType dsl_pipeline_main_loop_delete(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        PipelineMainLoopDelete(cstrName.c_str());
}

DslReturnType dsl_player_new(const wchar_t* name,
    const wchar_t* file_source, const wchar_t* sink)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_source);
    RETURN_IF_PARAM_IS_NULL(sink);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFileSource(file_source);
    std::string cstrFileSource(wstrFileSource.begin(), wstrFileSource.end());
    std::wstring wstrSink(sink);
    std::string cstrSink(wstrSink.begin(), wstrSink.end());

    return DSL::Services::GetServices()->PlayerNew(cstrName.c_str(),
        cstrFileSource.c_str(), cstrSink.c_str());
}

DslReturnType dsl_player_render_video_new(const wchar_t* name,  const wchar_t* file_path, 
   uint render_type, uint offset_x, uint offset_y, uint zoom, boolean repeat_enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrFilePath;
    if (file_path != NULL)
    {
        std::wstring wstrFilePath(file_path);
        cstrFilePath.assign(wstrFilePath.begin(), wstrFilePath.end());
    }

    return DSL::Services::GetServices()->PlayerRenderVideoNew(cstrName.c_str(),
        cstrFilePath.c_str(), render_type, offset_x, offset_y, zoom, repeat_enabled);
}

DslReturnType dsl_player_render_image_new(const wchar_t* name, const wchar_t* file_path,
    uint render_type, uint offset_x, uint offset_y, uint zoom, uint timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    std::string cstrFilePath;
    if (file_path != NULL)
    {
        std::wstring wstrFilePath(file_path);
        cstrFilePath.assign(wstrFilePath.begin(), wstrFilePath.end());
    }

    return DSL::Services::GetServices()->PlayerRenderImageNew(cstrName.c_str(),
        cstrFilePath.c_str(), render_type, offset_x, offset_y, zoom, timeout);
}

DslReturnType dsl_player_render_file_path_get(const wchar_t* name, 
    const wchar_t** file_path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cFilePath;
    static std::string cstrFilePath;
    static std::wstring wcstrFilePath;
    
    uint retval = DSL::Services::GetServices()->PlayerRenderFilePathGet(cstrName.c_str(), 
        &cFilePath);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrFilePath.assign(cFilePath);
        wcstrFilePath.assign(cstrFilePath.begin(), cstrFilePath.end());
        *file_path = wcstrFilePath.c_str();
    }
    return retval;
}

DslReturnType dsl_player_render_file_path_set(const wchar_t* name, 
    const wchar_t* file_path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->PlayerRenderFilePathSet(cstrName.c_str(), 
        cstrFilePath.c_str());
}

DslReturnType dsl_player_render_file_path_queue(const wchar_t* name, 
    const wchar_t* file_path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->PlayerRenderFilePathQueue(cstrName.c_str(), 
        cstrFilePath.c_str());
}

DslReturnType dsl_player_render_offsets_get(const wchar_t* name, uint* offset_x, uint* offset_y)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerRenderOffsetsGet(cstrName.c_str(), offset_x, offset_y);
}

DslReturnType dsl_player_render_offsets_set(const wchar_t* name, uint offset_x, uint offset_y)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerRenderOffsetsSet(cstrName.c_str(), offset_x, offset_y);
}

DslReturnType dsl_player_render_zoom_get(const wchar_t* name, uint* zoom)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(zoom);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerRenderZoomGet(cstrName.c_str(), zoom);
}

DslReturnType dsl_player_render_zoom_set(const wchar_t* name, uint zoom)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerRenderZoomSet(cstrName.c_str(), zoom);
}

DslReturnType dsl_player_render_reset(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerRenderReset(cstrName.c_str());
}

DslReturnType dsl_player_render_image_timeout_get(const wchar_t* name, uint* timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(timeout);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerRenderImageTimeoutGet(cstrName.c_str(), 
        timeout);
}

DslReturnType dsl_player_render_image_timeout_set(const wchar_t* name, uint timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerRenderImageTimeoutSet(cstrName.c_str(), 
        timeout);
}

DslReturnType dsl_player_render_video_repeat_enabled_get(const wchar_t* name, 
    boolean* repeat_enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(repeat_enabled);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerRenderVideoRepeatEnabledGet(cstrName.c_str(),
        repeat_enabled);
}
    
DslReturnType dsl_player_render_video_repeat_enabled_set(const wchar_t* name, 
    boolean repeat_enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerRenderVideoRepeatEnabledSet(cstrName.c_str(), 
        repeat_enabled);
}
    
DslReturnType dsl_player_termination_event_listener_add(const wchar_t* name, 
    dsl_player_termination_event_listener_cb listener, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        PlayerTerminationEventListenerAdd(cstrName.c_str(), listener, client_data);
}    

DslReturnType dsl_player_termination_event_listener_remove(const wchar_t* name, 
    dsl_player_termination_event_listener_cb listener)    
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        PlayerTerminationEventListenerRemove(cstrName.c_str(), listener);
}

DslReturnType dsl_player_play(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerPlay(cstrName.c_str());
}

DslReturnType dsl_player_pause(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerPause(cstrName.c_str());
}

DslReturnType dsl_player_stop(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerStop(cstrName.c_str());
}

DslReturnType dsl_player_render_next(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerRenderNext(cstrName.c_str());
}

DslReturnType dsl_player_state_get(const wchar_t* name, uint* state)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(state);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerStateGet(cstrName.c_str(), state);
}

boolean dsl_player_exists(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerExists(cstrName.c_str());
}

DslReturnType dsl_player_delete(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PlayerDelete(cstrName.c_str());
}

DslReturnType dsl_player_delete_all()
{
    return DSL::Services::GetServices()->PlayerDeleteAll();
}

uint dsl_player_list_size()
{
    return DSL::Services::GetServices()->PlayerListSize();
}

DslReturnType dsl_mailer_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->MailerNew(cstrName.c_str());
}

DslReturnType dsl_mailer_enabled_get(const wchar_t* name, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->MailerEnabledGet(cstrName.c_str(), 
        enabled);
}

DslReturnType dsl_mailer_enabled_set(const wchar_t* name, boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->MailerEnabledSet(cstrName.c_str(), 
        enabled);
}

DslReturnType dsl_mailer_credentials_set(const wchar_t* name, 
    const wchar_t* username, const wchar_t* password)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(username);
    RETURN_IF_PARAM_IS_NULL(password);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUsername(username);
    std::string cstrUsername(wstrUsername.begin(), wstrUsername.end());
    std::wstring wstrPassword(password);
    std::string cstrPassword(wstrPassword.begin(), wstrPassword.end());

    return DSL::Services::GetServices()->MailerCredentialsSet(cstrName.c_str(),
        cstrUsername.c_str(), cstrPassword.c_str());
}

DslReturnType dsl_mailer_server_url_get(const wchar_t* name, const wchar_t** server_url)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(server_url);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cServerUrl;
    static std::string cstrServerUrl;
    static std::wstring wcstrServerUrl;
    
    DslReturnType result = DSL::Services::GetServices()->MailerServerUrlGet(cstrName.c_str(),
        &cServerUrl);
    if (result == DSL_RESULT_SUCCESS)
    {
        cstrServerUrl.assign(cServerUrl);
        wcstrServerUrl.assign(cstrServerUrl.begin(), cstrServerUrl.end());
        *server_url = wcstrServerUrl.c_str();
    }
    return result;
}

DslReturnType dsl_mailer_server_url_set(const wchar_t* name, 
    const wchar_t* server_url)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(server_url);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrServerUrl(server_url);
    std::string cstrServerUrl(wstrServerUrl.begin(), wstrServerUrl.end());

    return DSL::Services::GetServices()->MailerServerUrlSet(cstrName.c_str(), 
        cstrServerUrl.c_str());
}

DslReturnType dsl_mailer_address_from_get(const wchar_t* name,
    const wchar_t** display_name, const wchar_t** address)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(display_name);
    RETURN_IF_PARAM_IS_NULL(address);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cDisplayName;
    static std::string cstrDisplayName;
    static std::wstring wcstrDisplayName;
    const char* cAddress;
    static std::string cstrAddress;
    static std::wstring wcstrAddress;
    
    DslReturnType result = DSL::Services::GetServices()->MailerFromAddressGet(
        cstrName.c_str(), &cDisplayName, &cAddress);
    if (result == DSL_RESULT_SUCCESS)
    {
        cstrDisplayName.assign(cDisplayName);
        wcstrDisplayName.assign(cstrDisplayName.begin(), cstrDisplayName.end());
        *display_name = wcstrDisplayName.c_str();
        cstrAddress.assign(cAddress);
        wcstrAddress.assign(cstrAddress.begin(), cstrAddress.end());
        *address = wcstrAddress.c_str();
    }
    return result;
}

DslReturnType dsl_mailer_address_from_set(const wchar_t* name,
    const wchar_t* display_name, const wchar_t* address)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(display_name);
    RETURN_IF_PARAM_IS_NULL(address);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrDisplayName(display_name);
    std::string cstrDisplayName(wstrDisplayName.begin(), wstrDisplayName.end());
    std::wstring wstrAddress(address);
    std::string cstrAddress(wstrAddress.begin(), wstrAddress.end());

    return DSL::Services::GetServices()->MailerFromAddressSet(cstrName.c_str(),
        cstrDisplayName.c_str(), cstrAddress.c_str());
}    

DslReturnType dsl_mailer_ssl_enabled_get(const wchar_t* name, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->MailerSslEnabledGet(
        cstrName.c_str(), enabled);
}

DslReturnType dsl_mailer_ssl_enabled_set(const wchar_t* name, boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->MailerSslEnabledSet(
        cstrName.c_str(), enabled);
}

DslReturnType dsl_mailer_address_to_add(const wchar_t* name,
    const wchar_t* display_name, const wchar_t* address)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(display_name);
    RETURN_IF_PARAM_IS_NULL(address);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrDisplayName(display_name);
    std::string cstrDisplayName(wstrDisplayName.begin(), wstrDisplayName.end());
    std::wstring wstrAddress(address);
    std::string cstrAddress(wstrAddress.begin(), wstrAddress.end());

    return DSL::Services::GetServices()->MailerToAddressAdd(cstrName.c_str(),
        cstrDisplayName.c_str(), cstrAddress.c_str());
}    
    
DslReturnType dsl_mailer_address_to_remove_all(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->MailerToAddressesRemoveAll(cstrName.c_str());
}
    
DslReturnType dsl_mailer_address_cc_add(const wchar_t* name,
    const wchar_t* display_name, const wchar_t* address)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(display_name);
    RETURN_IF_PARAM_IS_NULL(address);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrDisplayName(display_name);
    std::string cstrDisplayName(wstrDisplayName.begin(), wstrDisplayName.end());
    std::wstring wstrAddress(address);
    std::string cstrAddress(wstrAddress.begin(), wstrAddress.end());

    return DSL::Services::GetServices()->MailerCcAddressAdd(cstrName.c_str(),
        cstrDisplayName.c_str(), cstrAddress.c_str());
}    

DslReturnType dsl_mailer_address_cc_remove_all(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->MailerCcAddressesRemoveAll(cstrName.c_str());
}

DslReturnType dsl_mailer_test_message_send(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->MailerSendTestMessage(cstrName.c_str());
}    

boolean dsl_mailer_exists(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->MailerExists(cstrName.c_str());
}

DslReturnType dsl_mailer_delete(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->MailerDelete(cstrName.c_str());
}

DslReturnType dsl_mailer_delete_all()
{
    return DSL::Services::GetServices()->MailerDeleteAll();
}

uint dsl_mailer_list_size()
{
    return DSL::Services::GetServices()->MailerListSize();
}

DslReturnType dsl_message_broker_new(const wchar_t* name, 
    const wchar_t* broker_config_file, const wchar_t* protocol_lib, 
    const wchar_t* connection_string)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(broker_config_file);
    RETURN_IF_PARAM_IS_NULL(protocol_lib);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrBrokerConfig(broker_config_file);
    std::string cstrBrokerConfig(wstrBrokerConfig.begin(), wstrBrokerConfig.end());
    std::wstring wstrProtocolLib(protocol_lib);
    std::string cstrProtocolLib(wstrProtocolLib.begin(), wstrProtocolLib.end());
    
    std::string cstrConn;
    if (connection_string != NULL)
    {
        std::wstring wstrConn(connection_string);
        cstrConn.assign(wstrConn.begin(), wstrConn.end());
    }
    
    return DSL::Services::GetServices()->MessageBrokerNew(cstrName.c_str(), 
        cstrBrokerConfig.c_str(), cstrProtocolLib.c_str(), cstrConn.c_str());
}

DslReturnType dsl_message_broker_settings_get(const wchar_t* name, 
    const wchar_t** broker_config_file, const wchar_t**  protocol_lib,
    const wchar_t** connection_string)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(broker_config_file);
    RETURN_IF_PARAM_IS_NULL(protocol_lib);
    RETURN_IF_PARAM_IS_NULL(connection_string);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cConfigFile;
    static std::string cstrConfigFile;
    static std::wstring wcstrConfigFile;
    const char* cProtocolLib;
    static std::string cstrProtocolLib;
    static std::wstring wcstrProtocolLib;
    const char* cConnStr;
    static std::string cstrConnStr;
    static std::wstring wcstrConnStr;
    
    uint retval = DSL::Services::GetServices()->MessageBrokerSettingsGet(
        cstrName.c_str(), &cConfigFile, &cProtocolLib, &cConnStr);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrConfigFile.assign(cConfigFile);
        wcstrConfigFile.assign(cstrConfigFile.begin(), cstrConfigFile.end());
        *broker_config_file = wcstrConfigFile.c_str();
        cstrProtocolLib.assign(cProtocolLib);
        wcstrProtocolLib.assign(cstrProtocolLib.begin(), cstrProtocolLib.end());
        *protocol_lib = wcstrProtocolLib.c_str();
        cstrConnStr.assign(cConnStr);
        wcstrConnStr.assign(cstrConnStr.begin(), cstrConnStr.end());
        *connection_string = wcstrConnStr.c_str();
    }
    return retval;
} 

DslReturnType dsl_message_broker_settings_set(const wchar_t* name, 
    const wchar_t* broker_config_file, const wchar_t* protocol_lib,
    const wchar_t* connection_string)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(broker_config_file);
    RETURN_IF_PARAM_IS_NULL(protocol_lib);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrBrokerConfig(broker_config_file);
    std::string cstrBrokerConfig(wstrBrokerConfig.begin(), wstrBrokerConfig.end());
    std::wstring wstrProtocolLib(protocol_lib);
    std::string cstrProtocolLib(wstrProtocolLib.begin(), wstrProtocolLib.end());
    

    std::string cstrConn;
    if (connection_string != NULL)
    {
        std::wstring wstrConn(connection_string);
        cstrConn.assign(wstrConn.begin(), wstrConn.end());
    }
    
    return DSL::Services::GetServices()->MessageBrokerSettingsSet(cstrName.c_str(), 
        cstrBrokerConfig.c_str(), cstrProtocolLib.c_str(), 
        cstrConn.c_str());
}
    
DslReturnType dsl_message_broker_connect(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->MessageBrokerConnect(cstrName.c_str());
}

DslReturnType dsl_message_broker_disconnect(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->MessageBrokerDisconnect(cstrName.c_str());
}

DslReturnType dsl_message_broker_is_connected(const wchar_t* name,
    boolean* connected)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->MessageBrokerIsConnected(cstrName.c_str(),
        connected);
}

DslReturnType dsl_message_broker_message_send_async(const wchar_t* name,
    const wchar_t* topic, void* message, size_t size, 
    dsl_message_broker_send_result_listener_cb result_listener, void* user_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(result_listener);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    std::string cstrTopic;
    if (topic != NULL)
    {
        std::wstring wstrTopic(topic);
        cstrTopic.assign(wstrTopic.begin(), wstrTopic.end());
    }

    return DSL::Services::GetServices()->MessageBrokerMessageSendAsync(
        cstrName.c_str(), cstrTopic.c_str(), message, size, result_listener, user_data);
}
    
DslReturnType dsl_message_broker_subscriber_add(const wchar_t* name,
    dsl_message_broker_subscriber_cb subscriber, const wchar_t** topics,
    void* user_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(subscriber);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    std::vector<std::shared_ptr<std::string>> newTopics; 
    std::vector<const char*> cTopics;
    
    for (const wchar_t** topic = topics; *topic; topic++)
    {
        std::wstring wstrTopic(*topic);
        std::string cstrTopic(wstrTopic.begin(), wstrTopic.end());
        
        std::cout << "new topic = " << cstrTopic.c_str() << "\n";
        
        std::shared_ptr<std::string> newTopic = 
            std::shared_ptr<std::string>(new std::string(cstrTopic.c_str()));
        newTopics.push_back(newTopic);
        cTopics.push_back(newTopic->c_str());
    }
    cTopics.push_back(NULL);

    return DSL::Services::GetServices()->MessageBrokerSubscriberAdd(
        cstrName.c_str(), subscriber, &cTopics[0], newTopics.size(), user_data);
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_message_broker_subscriber_remove(const wchar_t* name,
    dsl_message_broker_subscriber_cb subscriber)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(subscriber);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->MessageBrokerSubscriberRemove(
        cstrName.c_str(), subscriber);
}

DslReturnType dsl_message_broker_connection_listener_add(const wchar_t* name,
    dsl_message_broker_connection_listener_cb handler, void* user_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->MessageBrokerConnectionListenerAdd(
        cstrName.c_str(), handler, user_data);
}

DslReturnType dsl_message_broker_connection_listener_remove(const wchar_t* name,
    dsl_message_broker_connection_listener_cb handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->MessageBrokerConnectionListenerRemove(
        cstrName.c_str(), handler);
}

DslReturnType dsl_message_broker_delete(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->MessageBrokerDelete(cstrName.c_str());
}

DslReturnType dsl_message_broker_delete_all()
{
    return DSL::Services::GetServices()->MessageBrokerDeleteAll();
}

uint dsl_message_broker_list_size()
{
    return DSL::Services::GetServices()->MessageBrokerListSize();
}
    
void dsl_delete_all()
{
    DSL::Services::GetServices()->DeleteAll();
}

const wchar_t* dsl_info_version_get()
{
    return DSL_VERSION;
}

boolean dsl_info_use_new_nvstreammux_get()
{
    return DSL::Services::GetServices()->UseNewStreammuxGet();
}

uint dsl_info_gpu_type_get(uint gpu_id)
{
    // Get the Device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, gpu_id);

    // if aarch64 build, set memorytype to default
    return (deviceProp.integrated) 
        ? DSL_GPU_TYPE_INTEGRATED
        : DSL_GPU_TYPE_DISCRETE;
}

DslReturnType dsl_info_stdout_get(const wchar_t** file_path)
{
    RETURN_IF_PARAM_IS_NULL(file_path);

    const char* cFilePath(NULL);
    static std::wstring wcstrFilePath;
    
    uint retval = DSL::Services::GetServices()->InfoStdoutGet(&cFilePath);
    if (retval ==  DSL_RESULT_SUCCESS and cFilePath)
    {
        std::string cstrFilePath(cFilePath);
        wcstrFilePath.assign(cstrFilePath.begin(), cstrFilePath.end());
        *file_path = wcstrFilePath.c_str();
    }
    return retval;
}


DslReturnType dsl_info_stdout_redirect(const wchar_t* file_path, uint mode)
{
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->InfoStdoutRedirect(
        cstrFilePath.c_str(), mode);
}

DslReturnType dsl_info_stdout_redirect_with_ts(const wchar_t* file_path)
{
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->InfoStdoutRedirectWithTs(
        cstrFilePath.c_str());
}

DslReturnType dsl_info_stdout_restore()
{
    return DSL::Services::GetServices()->InfoStdOutRestore();
}

DslReturnType dsl_info_log_level_get(const wchar_t** level)
{
    RETURN_IF_PARAM_IS_NULL(level);

    const char* cLevel(NULL);
    static std::wstring wcstrLevel;
    
    uint retval = DSL::Services::GetServices()->InfoLogLevelGet(&cLevel);
    if (retval ==  DSL_RESULT_SUCCESS and cLevel)
    {
        std::string cstrLevel(cLevel);
        wcstrLevel.assign(cstrLevel.begin(), cstrLevel.end());
        *level = wcstrLevel.c_str();
    }
    return retval;
}

DslReturnType dsl_info_log_level_set(const wchar_t* level)
{
    RETURN_IF_PARAM_IS_NULL(level);
    
    std::wstring wstrLevel(level);
    std::string cstrLevel(wstrLevel.begin(), wstrLevel.end());

    return DSL::Services::GetServices()->InfoLogLevelSet(cstrLevel.c_str());
}

DslReturnType dsl_info_log_file_get(const wchar_t** file_path)
{
    RETURN_IF_PARAM_IS_NULL(file_path);

    const char* cFilePath(NULL);
    static std::wstring wcstrFilePath;
    
    uint retval = DSL::Services::GetServices()->InfoLogFileGet(&cFilePath);
    if (retval ==  DSL_RESULT_SUCCESS and cFilePath)
    {
        std::string cstrFilePath(cFilePath);
        wcstrFilePath.assign(cstrFilePath.begin(), cstrFilePath.end());
        *file_path = wcstrFilePath.c_str();
    }
    return retval;
}

DslReturnType dsl_info_log_file_set(const wchar_t* file_path, uint mode)
{
    RETURN_IF_PARAM_IS_NULL(file_path);
    
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->InfoLogFileSet(
        cstrFilePath.c_str(), mode);
}

DslReturnType dsl_info_log_file_set_with_ts(const wchar_t* file_path)
{
    RETURN_IF_PARAM_IS_NULL(file_path);
    
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->InfoLogFileSetWithTs(cstrFilePath.c_str());
}

DslReturnType dsl_info_log_function_restore()
{
    return DSL::Services::GetServices()->InfoLogFunctionRestore();
}

DslReturnType dsl_setup_spd_logger(spdlog::logger* logger)
{
    return DSL::Services::GetServices()->SetSpdLogger(logger);
}