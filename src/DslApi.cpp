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
#include "DslServices.h"

#define RETURN_IF_PARAM_IS_NULL(input_string) do \
{ \
    if (!input_string) \
    { \
        LOG_ERROR("Input parameter must be a valid string and not NULL"); \
        return DSL_RESULT_INVALID_INPUT_PARAM; \
    } \
}while(0); 


DslReturnType dsl_display_type_rgba_color_new(const wchar_t* name, 
    double red, double green, double blue, double alpha)
{
    RETURN_IF_PARAM_IS_NULL(name);
    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaColorNew(cstrName.c_str(), 
        red, green, blue, alpha);
}

DslReturnType dsl_display_type_rgba_font_new(const wchar_t* name, const wchar_t* font, uint size, const wchar_t* color)
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

    return DSL::Services::GetServices()->DisplayTypeRgbaFontNew(cstrName.c_str(), cstrFont.c_str(),
        size, cstrColor.c_str());
}

DslReturnType dsl_display_type_rgba_text_new(const wchar_t* name, const wchar_t* text, 
    uint x_offset, uint y_offset, const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color)
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

    return DSL::Services::GetServices()->DisplayTypeRgbaTextNew(cstrName.c_str(), cstrText.c_str(),
        x_offset, y_offset, cstrFont.c_str(), has_bg_color, cstrBgColor.c_str());
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
    uint x1, uint y1, uint x2, uint y2, uint width, uint head, const wchar_t* color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(color);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaArrowNew(cstrName.c_str(), 
        x1, y1, x2, y2, width, head, cstrColor.c_str());
}
    
DslReturnType dsl_display_type_rgba_rectangle_new(const wchar_t* name, uint left, uint top, 
    uint width, uint height, uint border_width, const wchar_t* color, 
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

    return DSL::Services::GetServices()->DisplayTypeRgbaRectangleNew(cstrName.c_str(), 
        left, top, width, height, border_width, cstrColor.c_str(), has_bg_color, cstrBgColor.c_str());
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

    return DSL::Services::GetServices()->DisplayTypeRgbaPolygonNew(cstrName.c_str(), 
        coordinates, num_coordinates, border_width, cstrColor.c_str());
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

    return DSL::Services::GetServices()->DisplayTypeRgbaCircleNew(cstrName.c_str(), 
        x_center, y_center, radius, cstrColor.c_str(), has_bg_color, cstrBgColor.c_str());
}

DslReturnType dsl_display_type_source_number_new(const wchar_t* name, 
    uint x_offset, uint y_offset, const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color)
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

    return DSL::Services::GetServices()->DisplayTypeSourceNumberNew(cstrName.c_str(),
        x_offset, y_offset, cstrFont.c_str(), has_bg_color, cstrBgColor.c_str());
}

DslReturnType dsl_display_type_source_name_new(const wchar_t* name, 
    uint x_offset, uint y_offset, const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color)
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
    uint x_offset, uint y_offset, const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color)
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

    return DSL::Services::GetServices()->DisplayTypeSourceDimensionsNew(cstrName.c_str(),
        x_offset, y_offset, cstrFont.c_str(), has_bg_color, cstrBgColor.c_str());
}

// TODO: leaving this implementation as is without including in the header file for now.
// Needs to be completed and tested for all source types.
DslReturnType dsl_display_type_source_frame_rate_new(const wchar_t* name, 
    uint x_offset, uint y_offset, const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color)
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

    return DSL::Services::GetServices()->DisplayTypeSourceFrameRateNew(cstrName.c_str(),
        x_offset, y_offset, cstrFont.c_str(), has_bg_color, cstrBgColor.c_str());
}

DslReturnType dsl_display_type_meta_add(const wchar_t* name, void* display_meta, void* frame_meta)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTypeMetaAdd(cstrName.c_str(), display_meta, frame_meta);
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

DslReturnType dsl_ode_action_custom_new(const wchar_t* name, 
    dsl_ode_handle_occurrence_cb client_hanlder, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(client_hanlder);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionCustomNew(cstrName.c_str(), client_hanlder, client_data);
}

DslReturnType dsl_ode_action_capture_frame_new(const wchar_t* name, const wchar_t* outdir, boolean annotate)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(outdir);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOutdir(outdir);
    std::string cstrOutdir(wstrOutdir.begin(), wstrOutdir.end());

    return DSL::Services::GetServices()->OdeActionCaptureFrameNew(cstrName.c_str(), cstrOutdir.c_str(), annotate);
}

DslReturnType dsl_ode_action_capture_object_new(const wchar_t* name, const wchar_t* outdir)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(outdir);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOutdir(outdir);
    std::string cstrOutdir(wstrOutdir.begin(), wstrOutdir.end());

    return DSL::Services::GetServices()->OdeActionCaptureObjectNew(cstrName.c_str(), cstrOutdir.c_str());
}

DslReturnType dsl_ode_action_capture_complete_listener_add(const wchar_t* name, 
    dsl_capture_complete_listener_cb listener, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        OdeActionCaptureCompleteListenerAdd(cstrName.c_str(), listener, client_data);
}
    
DslReturnType dsl_ode_action_capture_complete_listener_remove(const wchar_t* name, 
    dsl_capture_complete_listener_cb listener)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->
        OdeActionCaptureCompleteListenerRemove(cstrName.c_str(), listener);
}
    
DslReturnType dsl_ode_action_display_new(const wchar_t* name, uint offset_x, uint offset_y, 
    boolean offset_y_with_classId, const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color)
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

    return DSL::Services::GetServices()->OdeActionDisplayNew(cstrName.c_str(),
        offset_x, offset_y, offset_y_with_classId, cstrFont.c_str(), has_bg_color, cstrBgColor.c_str());
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

DslReturnType dsl_ode_action_hide_new(const wchar_t* name, boolean text, boolean border)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionHideNew(cstrName.c_str(), text, border);
}

DslReturnType dsl_ode_action_email_new(const wchar_t* name, const wchar_t* subject)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(subject);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrSubject(subject);
    std::string cstrSubject(wstrSubject.begin(), wstrSubject.end());

    return DSL::Services::GetServices()->OdeActionEmailNew(cstrName.c_str(),
        cstrSubject.c_str());
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

DslReturnType dsl_ode_action_fill_object_new(const wchar_t* name, const wchar_t* color)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(color);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());

    return DSL::Services::GetServices()->OdeActionFillObjectNew(cstrName.c_str(),
        cstrColor.c_str());
}

DslReturnType dsl_ode_action_log_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionLogNew(cstrName.c_str());
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

    for (const wchar_t** display_type = display_types+1; *display_type; display_type++)
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

DslReturnType dsl_ode_action_pause_new(const wchar_t* name, const wchar_t* pipeline)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->OdeActionPauseNew(cstrName.c_str(), 
        cstrPipeline.c_str());
}

DslReturnType dsl_ode_action_print_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionPrintNew(cstrName.c_str());
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
    
DslReturnType dsl_ode_action_enabled_get(const wchar_t* name, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

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
    const wchar_t* line, boolean show, uint bbox_test_edge)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(line);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrLine(line);
    std::string cstrLine(wstrLine.begin(), wstrLine.end());

    return DSL::Services::GetServices()->OdeAreaLineNew(cstrName.c_str(), 
        cstrLine.c_str(), show, bbox_test_edge);
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

DslReturnType dsl_ode_trigger_occurrence_new(const wchar_t* name, const wchar_t* source, uint class_id, uint limit)
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
    return DSL::Services::GetServices()->OdeTriggerOccurrenceNew(cstrName.c_str(), cstrSource.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_absence_new(const wchar_t* name, const wchar_t* source, uint class_id, uint limit)
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
    return DSL::Services::GetServices()->OdeTriggerAbsenceNew(cstrName.c_str(), cstrSource.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_instance_new(const wchar_t* name, const wchar_t* source, uint class_id, uint limit)
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
    return DSL::Services::GetServices()->OdeTriggerInstanceNew(cstrName.c_str(), cstrSource.c_str(), class_id, limit);
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

DslReturnType dsl_ode_trigger_summation_new(const wchar_t* name, const wchar_t* source, uint class_id, uint limit)
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
    return DSL::Services::GetServices()->OdeTriggerPersistenceNew(cstrName.c_str(), cstrSource.c_str(), 
        class_id, limit, minimum, maximum);
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

DslReturnType dsl_ode_trigger_smallest_new(const wchar_t* name, const wchar_t* source, uint class_id, uint limit)
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
    return DSL::Services::GetServices()->OdeTriggerSmallestNew(cstrName.c_str(), cstrSource.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_largest_new(const wchar_t* name, const wchar_t* source, uint class_id, uint limit)
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
    return DSL::Services::GetServices()->OdeTriggerLargestNew(cstrName.c_str(), cstrSource.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_reset(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerReset(cstrName.c_str());
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

DslReturnType dsl_ode_trigger_limit_get(const wchar_t* name, uint* limit)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerLimitGet(cstrName.c_str(), limit);
}

DslReturnType dsl_ode_trigger_limit_set(const wchar_t* name, uint limit)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerLimitSet(cstrName.c_str(), limit);
}

DslReturnType dsl_ode_trigger_source_get(const wchar_t* name, const wchar_t** source)
{
    RETURN_IF_PARAM_IS_NULL(name);

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

DslReturnType dsl_ode_trigger_confidence_min_get(const wchar_t* name, float* min_confidence)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerConfidenceMinGet(cstrName.c_str(), min_confidence);
}

DslReturnType dsl_ode_trigger_confidence_min_set(const wchar_t* name, float min_confidence)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerConfidenceMinSet(cstrName.c_str(), min_confidence);
}

DslReturnType dsl_ode_trigger_dimensions_min_get(const wchar_t* name, float* min_width, float* min_height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDimensionsMinGet(cstrName.c_str(), min_width, min_height);
}

DslReturnType dsl_ode_trigger_dimensions_min_set(const wchar_t* name, float min_width, float min_height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDimensionsMinSet(cstrName.c_str(), min_width, min_height);
}

DslReturnType dsl_ode_trigger_dimensions_max_get(const wchar_t* name, float* max_width, float* max_height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDimensionsMaxGet(cstrName.c_str(), max_width, max_height);
}

DslReturnType dsl_ode_trigger_dimensions_max_set(const wchar_t* name, float max_width, float max_height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDimensionsMaxSet(cstrName.c_str(), max_width, max_height);
}

DslReturnType dsl_ode_trigger_infer_done_only_get(const wchar_t* name, boolean* infer_done_only)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerInferDoneOnlyGet(cstrName.c_str(), infer_done_only);
}

DslReturnType dsl_ode_trigger_infer_done_only_set(const wchar_t* name, boolean infer_done_only)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerInferDoneOnlySet(cstrName.c_str(), infer_done_only);
}

DslReturnType dsl_ode_trigger_frame_count_min_get(const wchar_t* name, uint* min_count_n, uint* min_count_d)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerFrameCountMinGet(cstrName.c_str(), min_count_n, min_count_d);
}

DslReturnType dsl_ode_trigger_frame_count_min_set(const wchar_t* name, uint min_count_n, uint min_count_d)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerFrameCountMinSet(cstrName.c_str(), min_count_n, min_count_d);
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

DslReturnType dsl_pph_custom_new(const wchar_t* name,
     dsl_pph_custom_client_handler_cb client_handler, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(client_handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->PphCustomNew(cstrName.c_str(), client_handler, client_data);
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

DslReturnType dsl_source_csi_new(const wchar_t* name, 
    uint width, uint height, uint fps_n, uint fps_d)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceCsiNew(cstrName.c_str(), 
        width, height, fps_n, fps_d);
}

DslReturnType dsl_source_usb_new(const wchar_t* name, 
    uint width, uint height, uint fps_n, uint fps_d)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceUsbNew(cstrName.c_str(), 
        width, height, fps_n, fps_d);
}

DslReturnType dsl_source_uri_new(const wchar_t* name, const wchar_t* uri, 
    boolean is_live, uint cudadec_mem_type, uint intra_decode, uint dropFrameInterval)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(uri);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUri(uri);
    std::string cstrUri(wstrUri.begin(), wstrUri.end());

    return DSL::Services::GetServices()->SourceUriNew(cstrName.c_str(), cstrUri.c_str(), 
        is_live, cudadec_mem_type, intra_decode, dropFrameInterval);
}

DslReturnType dsl_source_file_new(const wchar_t* name, 
    const wchar_t* file_path, boolean repeat_enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->SourceFileNew(cstrName.c_str(), 
        cstrFilePath.c_str(), repeat_enabled);
}

DslReturnType dsl_source_file_path_get(const wchar_t* name, 
    const wchar_t** file_path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cFilePath;
    static std::string cstrFilePath;
    static std::wstring wcstrFilePath;
    
    uint retval = DSL::Services::GetServices()->SourceFilePathGet(cstrName.c_str(), 
        &cFilePath);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrFilePath.assign(cFilePath);
        wcstrFilePath.assign(cstrFilePath.begin(), cstrFilePath.end());
        *file_path = wcstrFilePath.c_str();
    }
    return retval;
    
}

DslReturnType dsl_source_file_path_set(const wchar_t* name, const wchar_t* file_path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->SourceFilePathSet(cstrName.c_str(), 
        cstrFilePath.c_str());
}

DslReturnType dsl_source_file_repeat_enabled_get(const wchar_t* name, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceFileRepeatEnabledGet(cstrName.c_str(),
        enabled);
}

DslReturnType dsl_source_file_repeat_enabled_set(const wchar_t* name, boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceFileRepeatEnabledSet(cstrName.c_str(),
        enabled);
}

DslReturnType dsl_source_image_new(const wchar_t* name, 
    const wchar_t* file_path, boolean is_live, uint fps_n, uint fps_d, uint timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->SourceImageNew(cstrName.c_str(), 
        cstrFilePath.c_str(), is_live, fps_n, fps_d, timeout);
}

DslReturnType dsl_source_image_path_get(const wchar_t* name, 
    const wchar_t** file_path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cFilePath;
    static std::string cstrFilePath;
    static std::wstring wcstrFilePath;
    
    uint retval = DSL::Services::GetServices()->SourceImagePathGet(cstrName.c_str(), 
        &cFilePath);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrFilePath.assign(cFilePath);
        wcstrFilePath.assign(cstrFilePath.begin(), cstrFilePath.end());
        *file_path = wcstrFilePath.c_str();
    }
    return retval;
    
}

DslReturnType dsl_source_image_path_set(const wchar_t* name, const wchar_t* file_path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->SourceImagePathSet(cstrName.c_str(), 
        cstrFilePath.c_str());
}

DslReturnType dsl_source_image_timeout_get(const wchar_t* name, uint* timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceImageTimeoutGet(cstrName.c_str(),
        timeout);
}

DslReturnType dsl_source_image_timeout_set(const wchar_t* name, uint timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceImageTimeoutSet(cstrName.c_str(),
        timeout);
}

DslReturnType dsl_source_rtsp_new(const wchar_t* name, const wchar_t* uri, uint protocol, 
    uint cudadec_mem_type, uint intra_decode, uint dropFrameInterval, uint latency, uint timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(uri);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUri(uri);
    std::string cstrUri(wstrUri.begin(), wstrUri.end());

    return DSL::Services::GetServices()->SourceRtspNew(cstrName.c_str(), cstrUri.c_str(), 
        protocol, cudadec_mem_type, intra_decode, dropFrameInterval, latency, timeout);
}

DslReturnType dsl_source_dimensions_get(const wchar_t* name, uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceDimensionsGet(cstrName.c_str(), width, height);
}

DslReturnType dsl_source_frame_rate_get(const wchar_t* name, uint* fps_n, uint* fps_d)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceFrameRateGet(cstrName.c_str(), fps_n, fps_d);
}

DslReturnType dsl_source_decode_uri_get(const wchar_t* name, const wchar_t** uri)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(uri);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cUri;
    static std::string cstrUri;
    static std::wstring wcstrUri;
    
    uint retval = DSL::Services::GetServices()->SourceDecodeUriGet(cstrName.c_str(), &cUri);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrUri.assign(cUri);
        wcstrUri.assign(cstrUri.begin(), cstrUri.end());
        *uri = wcstrUri.c_str();
    }
    return retval;
}

DslReturnType dsl_source_decode_uri_set(const wchar_t* name, const wchar_t* uri)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(uri);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUri(uri);
    std::string cstrUri(wstrUri.begin(), wstrUri.end());

    return DSL::Services::GetServices()->SourceDecodeUriSet(cstrName.c_str(), cstrUri.c_str());
}

DslReturnType dsl_source_decode_dewarper_add(const wchar_t* name, const wchar_t* dewarper)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(dewarper);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrDewarper(dewarper);
    std::string cstrDewarper(wstrDewarper.begin(), wstrDewarper.end());

    return DSL::Services::GetServices()->SourceDecodeDewarperAdd(cstrName.c_str(), cstrDewarper.c_str());
}

DslReturnType dsl_source_decode_dewarper_remove(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceDecodeDewarperRemove(cstrName.c_str());
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

DslReturnType dsl_source_rtsp_reconnection_params_get(const wchar_t* name, uint* sleep, uint* timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceRtspReconnectionParamsGet(cstrName.c_str(), sleep, timeout);
}

DslReturnType dsl_source_rtsp_reconnection_params_set(const wchar_t* name, uint sleep, uint timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceRtspReconnectionParamsSet(cstrName.c_str(), sleep, timeout);
}

DslReturnType dsl_source_rtsp_connection_data_get(const wchar_t* name, dsl_rtsp_connection_data* data)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SourceRtspConnectionDataGet(cstrName.c_str(), data);
}

DslReturnType dsl_source_rtsp_connection_stats_clear(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceRtspConnectionStatsClear(cstrName.c_str());
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

DslReturnType dsl_source_name_get(uint source_id, const wchar_t** name)
{
    const char* cName;
    static std::string cstrName;
    static std::wstring wcstrName;
    
    uint retval = DSL::Services::GetServices()->SourceNameGet(source_id, &cName);
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

uint dsl_source_num_in_use_get()
{
    return DSL::Services::GetServices()->SourceNumInUseGet();
}

uint dsl_source_num_in_use_max_get()
{
    return DSL::Services::GetServices()->SourceNumInUseMaxGet();
}

boolean dsl_source_num_in_use_max_set(uint max)
{
    return DSL::Services::GetServices()->SourceNumInUseMaxSet(max);
}

DslReturnType dsl_dewarper_new(const wchar_t* name, const wchar_t* config_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());

    return DSL::Services::GetServices()->DewarperNew(cstrName.c_str(), cstrConfig.c_str());
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

DslReturnType dsl_tap_record_session_stop(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TapRecordSessionStop(cstrName.c_str());
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

DslReturnType dsl_gie_primary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* model_engine_file, uint interval)
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
    return DSL::Services::GetServices()->PrimaryGieNew(cstrName.c_str(), cstrConfig.c_str(),
        cstrEngine.c_str(), interval);
}

DslReturnType dsl_gie_primary_pph_add(const wchar_t* name, 
    const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->PrimaryGiePphAdd(cstrName.c_str(), cstrHandler.c_str(), pad);
}

DslReturnType dsl_gie_primary_pph_remove(const wchar_t* name,
    const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->PrimaryGiePphRemove(cstrName.c_str(), cstrHandler.c_str(), pad);
}

DslReturnType dsl_gie_secondary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* model_engine_file, const wchar_t* infer_on_gie, uint interval)
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
    return DSL::Services::GetServices()->SecondaryGieNew(cstrName.c_str(), cstrConfig.c_str(),
        cstrEngine.c_str(), cstrInferOnGie.c_str(), interval);
}

DslReturnType dsl_gie_infer_config_file_get(const wchar_t* name, const wchar_t** infer_config_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(infer_config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cConfig;
    static std::string cstrConfig;
    static std::wstring wcstrConfig;
    
    uint retval = DSL::Services::GetServices()->GieInferConfigFileGet(cstrName.c_str(), &cConfig);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrConfig.assign(cConfig);
        wcstrConfig.assign(cstrConfig.begin(), cstrConfig.end());
        *infer_config_file = wcstrConfig.c_str();
    }
    return retval;
}

DslReturnType dsl_gie_infer_config_file_set(const wchar_t* name, const wchar_t* infer_config_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(infer_config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(infer_config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());

    return DSL::Services::GetServices()->GieInferConfigFileSet(cstrName.c_str(), cstrConfig.c_str());
}

DslReturnType dsl_gie_model_engine_file_get(const wchar_t* name, const wchar_t** model_engine_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(model_engine_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cEngine;
    static std::string cstrEngine;
    static std::wstring wcstrEngine;
    
    uint retval = DSL::Services::GetServices()->GieModelEngineFileGet(cstrName.c_str(), &cEngine);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrEngine.assign(cEngine);
        wcstrEngine.assign(cstrEngine.begin(), cstrEngine.end());
        *model_engine_file = wcstrEngine.c_str();
    }
    return retval;
}

DslReturnType dsl_gie_model_engine_file_set(const wchar_t* name, const wchar_t* model_engine_file)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(model_engine_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrEngine(model_engine_file);
    std::string cstrEngine(wstrEngine.begin(), wstrEngine.end());

    return DSL::Services::GetServices()->GieModelEngineFileSet(cstrName.c_str(), cstrEngine.c_str());
}

DslReturnType dsl_gie_interval_get(const wchar_t* name, uint* interval)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->GieIntervalGet(cstrName.c_str(), interval);
}

DslReturnType dsl_gie_interval_set(const wchar_t* name, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->GieIntervalSet(cstrName.c_str(), interval);
}


DslReturnType dsl_gie_raw_output_enabled_set(const wchar_t* name, boolean enabled, const wchar_t* path)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPath(path);
    std::string cstrPath(wstrPath.begin(), wstrPath.end());

    return DSL::Services::GetServices()->GieRawOutputEnabledSet(cstrName.c_str(), enabled, cstrPath.c_str());
}

DslReturnType dsl_tracker_ktl_new(const wchar_t* name, uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TrackerKtlNew(cstrName.c_str(), width, height);
}
    
DslReturnType dsl_tracker_iou_new(const wchar_t* name, const wchar_t* config_file, uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(config_file);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFile(config_file);
    std::string cstrFile(wstrFile.begin(), wstrFile.end());

    return DSL::Services::GetServices()->TrackerIouNew(cstrName.c_str(), cstrFile.c_str(), width, height);
}

DslReturnType dsl_tracker_max_dimensions_get(const wchar_t* name, uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TrackerMaxDimensionsGet(cstrName.c_str(), width, height);
}

DslReturnType dsl_tracker_max_dimensions_set(const wchar_t* name, uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TrackerMaxDimensionsSet(cstrName.c_str(), width, height);
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
    
    return DSL::Services::GetServices()->TrackerPphAdd(cstrName.c_str(), cstrHandler.c_str(), pad);
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
    
    return DSL::Services::GetServices()->TrackerPphRemove(cstrName.c_str(), cstrHandler.c_str(), pad);
}

DslReturnType dsl_ofv_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OfvNew(cstrName.c_str());
}

DslReturnType dsl_osd_new(const wchar_t* name, 
    boolean text_enabled, boolean clock_enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdNew(cstrName.c_str(), 
        text_enabled, clock_enabled);
}

DslReturnType dsl_osd_text_enabled_get(const wchar_t* name, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(name);

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

DslReturnType dsl_osd_clock_offsets_get(const wchar_t* name, uint* offset_x, uint* offset_y)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockOffsetsGet(cstrName.c_str(), offset_x, offset_y);
}

DslReturnType dsl_osd_clock_offsets_set(const wchar_t* name, uint offset_x, uint offset_y)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockOffsetsSet(cstrName.c_str(), offset_x, offset_y);
}

DslReturnType dsl_osd_clock_font_get(const wchar_t* name, const wchar_t** font, uint* size)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    const char* cfont;
    static std::string cstrFont;
    static std::wstring wcstrFont;
    
    uint retval = DSL::Services::GetServices()->OsdClockFontGet(cstrName.c_str(), &cfont, size);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrFont.assign(cfont);
        wcstrFont.assign(cstrFont.begin(), cstrFont.end());
        *font = wcstrFont.c_str();
    }
    return retval;
}

DslReturnType dsl_osd_clock_font_set(const wchar_t* name, const wchar_t* font, uint size)
{
    std::cout << "***** pre services" << "\n";
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(font);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFont(font);
    std::string cstrFont(wstrFont.begin(), wstrFont.end());

    return DSL::Services::GetServices()->OsdClockFontSet(cstrName.c_str(), cstrFont.c_str(), size);
}

DslReturnType dsl_osd_clock_color_get(const wchar_t* name, double* red, double* green, double* blue, double* alpha)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockColorGet(cstrName.c_str(), red, green, blue, alpha);
}

DslReturnType dsl_osd_clock_color_set(const wchar_t* name, double red, double green, double blue, double alpha)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockColorSet(cstrName.c_str(), red, green, blue, alpha);
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
    
    return DSL::Services::GetServices()->OsdPphAdd(cstrName.c_str(), cstrHandler.c_str(), pad);
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
    
    return DSL::Services::GetServices()->OsdPphRemove(cstrName.c_str(), cstrHandler.c_str(), pad);
}

DslReturnType dsl_tee_demuxer_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TeeDemuxerNew(cstrName.c_str());
}

DslReturnType dsl_tee_demuxer_new_branch_add_many(const wchar_t* name, const wchar_t** branches)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branches);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    DslReturnType retval = DSL::Services::GetServices()->TeeDemuxerNew(cstrName.c_str());
    if (retval != DSL_RESULT_SUCCESS)
    {
        return retval;
    }

    for (const wchar_t** branch = branches; *branch; branch++)
    {
        std::wstring wstrBranch(*branch);
        std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        retval = DSL::Services::GetServices()->TeeBranchAdd(cstrName.c_str(), cstrBranch.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_tee_splitter_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TeeSplitterNew(cstrName.c_str());
}

DslReturnType dsl_tee_splitter_new_branch_add_many(const wchar_t* name, const wchar_t** branches)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branches);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    DslReturnType retval = DSL::Services::GetServices()->TeeSplitterNew(cstrName.c_str());
    if (retval != DSL_RESULT_SUCCESS)
    {
        return retval;
    }

    for (const wchar_t** branch = branches; *branch; branch++)
    {
        std::wstring wstrBranch(*branch);
        std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        retval = DSL::Services::GetServices()->TeeBranchAdd(cstrName.c_str(), cstrBranch.c_str());
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

    return DSL::Services::GetServices()->TeeBranchAdd(cstrName.c_str(), cstrBranch.c_str());
}

DslReturnType dsl_tee_branch_add_many(const wchar_t* name, const wchar_t** branches)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branches);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    for (const wchar_t** branch = branches; *branch; branch++)
    {
        std::wstring wstrBranch(*branch);
        std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        DslReturnType retval = DSL::Services::GetServices()->TeeBranchAdd(cstrName.c_str(), cstrBranch.c_str());
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

    return DSL::Services::GetServices()->TeeBranchRemove(cstrName.c_str(), cstrBranch.c_str());
}

DslReturnType dsl_tee_branch_remove_many(const wchar_t* name, const wchar_t** branches)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(branches);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    for (const wchar_t** branch = branches; *branch; branch++)
    {
        std::wstring wstrBranch(*branch);
        std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        DslReturnType retval = DSL::Services::GetServices()->TeeBranchRemove(cstrName.c_str(), cstrBranch.c_str());
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

DslReturnType dsl_tee_pph_add(const wchar_t* name, const wchar_t* handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->TeePphAdd(cstrName.c_str(), cstrHandler.c_str());
}

DslReturnType dsl_tee_pph_remove(const wchar_t* name, const wchar_t* handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->TeePphRemove(cstrName.c_str(), cstrHandler.c_str());
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

DslReturnType dsl_tiler_source_show_get(const wchar_t* name, 
    const wchar_t** source, uint* timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(source);

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

DslReturnType dsl_sink_fake_new(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkFakeNew(cstrName.c_str());
}

DslReturnType dsl_sink_overlay_new(const wchar_t* name, uint display_id,
    uint depth, uint offset_x, uint offset_y, uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkOverlayNew(cstrName.c_str(), 
        display_id, depth, offset_x, offset_y, width, height);
}

DslReturnType dsl_sink_window_new(const wchar_t* name,
    uint offset_x, uint offset_y, uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowNew(cstrName.c_str(), 
        offset_x, offset_y, width, height);
}

DslReturnType dsl_sink_window_force_aspect_ratio_get(const wchar_t* name, 
    boolean* force)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(force);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowForceAspectRationGet(cstrName.c_str(), 
        force);
}
    
DslReturnType dsl_sink_window_force_aspect_ratio_set(const wchar_t* name, 
    boolean force)    
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowForceAspectRationSet(cstrName.c_str(), 
        force);
}

DslReturnType dsl_sink_render_offsets_get(const wchar_t* name, uint* offset_x, uint* offset_y)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRenderOffsetsGet(cstrName.c_str(), offset_x, offset_y);
}

DslReturnType dsl_sink_render_offsets_set(const wchar_t* name, uint offset_x, uint offset_y)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRenderOffsetsSet(cstrName.c_str(), offset_x, offset_y);
}

DslReturnType dsl_sink_render_dimensions_get(const wchar_t* name, uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRenderDimensionsGet(cstrName.c_str(), width, height);
}

DslReturnType dsl_sink_render_dimensions_set(const wchar_t* name, uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRenderDimensionsSet(cstrName.c_str(), width, height);
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

DslReturnType dsl_sink_encode_video_formats_get(const wchar_t* name,
    uint* codec, uint* container)
{    
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkEncodeVideoFormatsGet(cstrName.c_str(), 
        codec, container);
}

DslReturnType dsl_sink_encode_settings_get(const wchar_t* name,
    uint* bitrate, uint* interval)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkEncodeSettingsGet(cstrName.c_str(), 
        bitrate, interval);
}    

DslReturnType dsl_sink_encode_settings_set(const wchar_t* name,
    uint bitrate, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkEncodeSettingsSet(cstrName.c_str(), 
        bitrate, interval);
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

DslReturnType dsl_sink_record_session_stop(const wchar_t* name)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordSessionStop(cstrName.c_str());
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
   
DslReturnType dsl_sink_rtsp_new(const wchar_t* name, const wchar_t* host, 
     uint udpPort, uint rtspPort, uint codec, uint bitrate, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(host);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHost(host);
    std::string cstrHost(wstrHost.begin(), wstrHost.end());

    return DSL::Services::GetServices()->SinkRtspNew(cstrName.c_str(), 
        cstrHost.c_str(), udpPort, rtspPort, codec, bitrate, interval);
}     

DslReturnType dsl_sink_rtsp_server_settings_get(const wchar_t* name,
    uint* udpPort, uint* rtspPort, uint* codec)
{    
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkRtspServerSettingsGet(cstrName.c_str(), 
        udpPort, rtspPort, codec);
}    

DslReturnType dsl_sink_rtsp_encoder_settings_get(const wchar_t* name,
    uint* bitrate, uint* interval)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkRtspEncoderSettingsGet(cstrName.c_str(), bitrate, interval);
}    

DslReturnType dsl_sink_rtsp_encoder_settings_set(const wchar_t* name,
    uint bitrate, uint interval)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkRtspEncoderSettingsSet(cstrName.c_str(), bitrate, interval);
}

DslReturnType dsl_sink_pph_add(const wchar_t* name, const wchar_t* handler)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->SinkPphAdd(cstrName.c_str(), cstrHandler.c_str());
}

DslReturnType dsl_sink_pph_remove(const wchar_t* name,
    const wchar_t* handler, uint pad)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());
    
    return DSL::Services::GetServices()->SinkPphRemove(cstrName.c_str(), cstrHandler.c_str());
}

DslReturnType dsl_sink_sync_settings_get(const wchar_t* name, boolean* sync, boolean* async)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkSyncSettingsGet(cstrName.c_str(), sync, async);
}
    
DslReturnType dsl_sink_sync_settings_set(const wchar_t* name, boolean sync, boolean async)
{
    RETURN_IF_PARAM_IS_NULL(name);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkSyncSettingsSet(cstrName.c_str(), sync, async);
}
    
uint dsl_sink_num_in_use_get()
{
    return DSL::Services::GetServices()->SinkNumInUseGet();
}

uint dsl_sink_num_in_use_max_get()
{
    return DSL::Services::GetServices()->SinkNumInUseMaxGet();
}

boolean dsl_sink_num_in_use_max_set(uint max)
{
    return DSL::Services::GetServices()->SinkNumInUseMaxSet(max);
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
        DslReturnType retval = DSL::Services::GetServices()->ComponentGpuIdSet(cstrName.c_str(), gpuid);
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

DslReturnType dsl_pipeline_new(const wchar_t* pipeline)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineNew(cstrPipeline.c_str());
}

DslReturnType dsl_pipeline_new_component_add_many(const wchar_t* pipeline, 
    const wchar_t** components)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(components);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
    
    DslReturnType retval = DSL::Services::GetServices()->PipelineNew(cstrPipeline.c_str());
    if (retval != DSL_RESULT_SUCCESS)
    {
        return retval;
    }
    for (const wchar_t** component = components; *component; component++)
    {
        std::wstring wstrComponent(*component);
        std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());
        DslReturnType retval = DSL::Services::GetServices()->PipelineComponentAdd(cstrPipeline.c_str(), cstrComponent.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_pipeline_new_many(const wchar_t** pipelines)
{
    RETURN_IF_PARAM_IS_NULL(pipelines);

    for (const wchar_t** pipeline = pipelines; *pipeline; pipeline++)
    {
        std::wstring wstrPipeline(*pipeline);
        std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
        DslReturnType retval = DSL::Services::GetServices()->PipelineNew(cstrPipeline.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_pipeline_delete(const wchar_t* pipeline)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineDelete(cstrPipeline.c_str());
}

DslReturnType dsl_pipeline_delete_many(const wchar_t** pipelines)
{
    RETURN_IF_PARAM_IS_NULL(pipelines);

    for (const wchar_t** pipeline = pipelines; *pipeline; pipeline++)
    {
        std::wstring wstrPipeline(*pipeline);
        std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
        DslReturnType retval = DSL::Services::GetServices()->PipelineDelete(cstrPipeline.c_str());
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

DslReturnType dsl_pipeline_component_add(const wchar_t* pipeline, 
    const wchar_t* component)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(component);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
    std::wstring wstrComponent(component);
    std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());

    return DSL::Services::GetServices()->PipelineComponentAdd(cstrPipeline.c_str(), cstrComponent.c_str());
}

DslReturnType dsl_pipeline_component_add_many(const wchar_t* pipeline, 
    const wchar_t** components)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(components);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
    
    for (const wchar_t** component = components; *component; component++)
    {
        std::wstring wstrComponent(*component);
        std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());
        DslReturnType retval = DSL::Services::GetServices()->PipelineComponentAdd(cstrPipeline.c_str(), cstrComponent.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_pipeline_component_remove(const wchar_t* pipeline, 
    const wchar_t* component)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(component);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
    std::wstring wstrComponent(component);
    std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());

    return DSL::Services::GetServices()->PipelineComponentRemove(cstrPipeline.c_str(), cstrComponent.c_str());
}

DslReturnType dsl_pipeline_component_remove_many(const wchar_t* pipeline, 
    const wchar_t** components)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(components);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
    
    for (const wchar_t** component = components; *component; component++)
    {
        std::wstring wstrComponent(*component);
        std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());
        DslReturnType retval = DSL::Services::GetServices()->PipelineComponentRemove(cstrPipeline.c_str(), cstrComponent.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_pipeline_streammux_batch_properties_get(const wchar_t* pipeline, 
    uint* batchSize, uint* batchTimeout)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(batchSize);
    RETURN_IF_PARAM_IS_NULL(batchTimeout);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStreamMuxBatchPropertiesGet(cstrPipeline.c_str(),
        batchSize, batchTimeout);
}

DslReturnType dsl_pipeline_streammux_batch_properties_set(const wchar_t* pipeline, 
    uint batchSize, uint batchTimeout)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStreamMuxBatchPropertiesSet(cstrPipeline.c_str(),
        batchSize, batchTimeout);
}

DslReturnType dsl_pipeline_streammux_dimensions_get(const wchar_t* pipeline, 
    uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(width);
    RETURN_IF_PARAM_IS_NULL(height);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStreamMuxDimensionsGet(cstrPipeline.c_str(),
        width, height);
}

DslReturnType dsl_pipeline_streammux_dimensions_set(const wchar_t* pipeline, 
    uint width, uint height)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStreamMuxDimensionsSet(cstrPipeline.c_str(),
        width, height);
}    

DslReturnType dsl_pipeline_streammux_padding_get(const wchar_t* pipeline, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStreamMuxPaddingGet(cstrPipeline.c_str(), enabled);
}

DslReturnType dsl_pipeline_streammux_padding_set(const wchar_t* pipeline, boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStreamMuxPaddingSet(cstrPipeline.c_str(), enabled);
}

DslReturnType dsl_pipeline_streammux_num_surfaces_per_frame_get(const wchar_t* pipeline, uint* num)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(num);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStreamMuxNumSurfacesPerFrameGet(cstrPipeline.c_str(), num);
}

DslReturnType dsl_pipeline_streammux_num_surfaces_per_frame_set(const wchar_t* pipeline, uint num)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStreamMuxNumSurfacesPerFrameSet(cstrPipeline.c_str(), num);
}

DslReturnType dsl_pipeline_xwindow_handle_get(const wchar_t* pipeline, uint64_t* xwindow)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(xwindow);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineXWindowHandleGet(cstrPipeline.c_str(), xwindow);
}

DslReturnType dsl_pipeline_xwindow_handle_set(const wchar_t* pipeline, uint64_t xwindow)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineXWindowHandleSet(cstrPipeline.c_str(), xwindow);
}

DslReturnType dsl_pipeline_xwindow_clear(const wchar_t* pipeline)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineXWindowClear(cstrPipeline.c_str());
}
 
DslReturnType dsl_pipeline_xwindow_destroy(const wchar_t* pipeline)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineXWindowDestroy(cstrPipeline.c_str());
}
 
DslReturnType dsl_pipeline_xwindow_offsets_get(const wchar_t* pipeline, 
    uint* x_offset, uint* y_offset)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(x_offset);
    RETURN_IF_PARAM_IS_NULL(y_offset);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineXWindowOffsetsGet(cstrPipeline.c_str(),
        x_offset, y_offset);
}

DslReturnType dsl_pipeline_xwindow_dimensions_get(const wchar_t* pipeline, 
    uint* width, uint* height)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(width);
    RETURN_IF_PARAM_IS_NULL(height);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineXWindowDimensionsGet(cstrPipeline.c_str(),
        width, height);
}

DslReturnType dsl_pipeline_xwindow_fullscreen_enabled_get(const wchar_t* pipeline, boolean* enabled)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(enabled);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineXWindowFullScreenEnabledGet(cstrPipeline.c_str(), enabled);
}

DslReturnType dsl_pipeline_xwindow_fullscreen_enabled_set(const wchar_t* pipeline, boolean enabled)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineXWindowFullScreenEnabledSet(cstrPipeline.c_str(), enabled);
}

DslReturnType dsl_pipeline_pause(const wchar_t* pipeline)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelinePause(cstrPipeline.c_str());
}

DslReturnType dsl_pipeline_play(const wchar_t* pipeline)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelinePlay(cstrPipeline.c_str());
}

DslReturnType dsl_pipeline_stop(const wchar_t* pipeline)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStop(cstrPipeline.c_str());
}

DslReturnType dsl_pipeline_state_get(const wchar_t* pipeline, uint* state)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(state);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStateGet(cstrPipeline.c_str(), state);
}

DslReturnType dsl_pipeline_is_live(const wchar_t* pipeline, boolean* is_live)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(is_live);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineIsLive(cstrPipeline.c_str(), is_live);
}

DslReturnType dsl_pipeline_dump_to_dot(const wchar_t* pipeline, wchar_t* filename)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(filename);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
    std::wstring wstrFilename(filename);
    std::string cstrFilename(wstrFilename.begin(), wstrFilename.end());

    return DSL::Services::GetServices()->PipelineDumpToDot(cstrPipeline.c_str(), 
        const_cast<char*>(cstrFilename.c_str()));
}

DslReturnType dsl_pipeline_dump_to_dot_with_ts(const wchar_t* pipeline, wchar_t* filename)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(filename);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
    std::wstring wstrFilename(filename);
    std::string cstrFilename(wstrFilename.begin(), wstrFilename.end());

    return DSL::Services::GetServices()->PipelineDumpToDotWithTs(cstrPipeline.c_str(), 
        const_cast<char*>(cstrFilename.c_str()));
}

DslReturnType dsl_pipeline_state_change_listener_add(const wchar_t* pipeline, 
    dsl_state_change_listener_cb listener, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineStateChangeListenerAdd(cstrPipeline.c_str(), listener, client_data);
}

DslReturnType dsl_pipeline_state_change_listener_remove(const wchar_t* pipeline, 
    dsl_state_change_listener_cb listener)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineStateChangeListenerRemove(cstrPipeline.c_str(), listener);
}

DslReturnType dsl_pipeline_eos_listener_add(const wchar_t* pipeline, 
    dsl_eos_listener_cb listener, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineEosListenerAdd(cstrPipeline.c_str(), listener, client_data);
}

DslReturnType dsl_pipeline_eos_listener_remove(const wchar_t* pipeline, 
    dsl_eos_listener_cb listener)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(listener);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineEosListenerRemove(cstrPipeline.c_str(), listener);
}

DslReturnType dsl_pipeline_error_message_handler_add(const wchar_t* pipeline, 
    dsl_error_message_handler_cb handler, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineErrorMessageHandlerAdd(cstrPipeline.c_str(), handler, client_data);
}

DslReturnType dsl_pipeline_error_message_handler_remove(const wchar_t* pipeline, 
    dsl_error_message_handler_cb handler)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineErrorMessageHandlerRemove(cstrPipeline.c_str(), handler);
}

DslReturnType dsl_pipeline_error_message_last_get(const wchar_t* pipeline, 
    const wchar_t** source, const wchar_t** message)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(source);
    RETURN_IF_PARAM_IS_NULL(message);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
    
    static std::wstring wstrSource;
    static std::wstring wstrMessage;
    
    uint retval = DSL::Services::GetServices()->PipelineErrorMessageLastGet(cstrPipeline.c_str(), wstrSource, wstrMessage);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        
        *source = (wstrSource.size()) ? wstrSource.c_str() : NULL;
        *message = (wstrMessage.size()) ? wstrMessage.c_str() : NULL;
    }
    return retval;
}
    
DslReturnType dsl_pipeline_xwindow_key_event_handler_add(const wchar_t* pipeline, 
    dsl_xwindow_key_event_handler_cb handler, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineXWindowKeyEventHandlerAdd(cstrPipeline.c_str(), handler, client_data);
}    

DslReturnType dsl_pipeline_xwindow_key_event_handler_remove(const wchar_t* pipeline, 
    dsl_xwindow_key_event_handler_cb handler)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineXWindowKeyEventHandlerRemove(cstrPipeline.c_str(), handler);
}

DslReturnType dsl_pipeline_xwindow_button_event_handler_add(const wchar_t* pipeline, 
    dsl_xwindow_button_event_handler_cb handler, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineXWindowButtonEventHandlerAdd(cstrPipeline.c_str(), handler, client_data);
}    

DslReturnType dsl_pipeline_xwindow_button_event_handler_remove(const wchar_t* pipeline, 
    dsl_xwindow_button_event_handler_cb handler)    
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineXWindowButtonEventHandlerRemove(cstrPipeline.c_str(), handler);
}

DslReturnType dsl_pipeline_xwindow_delete_event_handler_add(const wchar_t* pipeline, 
    dsl_xwindow_delete_event_handler_cb handler, void* client_data)
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineXWindowDeleteEventHandlerAdd(cstrPipeline.c_str(), handler, client_data);
}    

DslReturnType dsl_pipeline_xwindow_delete_event_handler_remove(const wchar_t* pipeline, 
    dsl_xwindow_delete_event_handler_cb handler)    
{
    RETURN_IF_PARAM_IS_NULL(pipeline);
    RETURN_IF_PARAM_IS_NULL(handler);

    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineXWindowDeleteEventHandlerRemove(cstrPipeline.c_str(), handler);
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
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

    return DSL::Services::GetServices()->PlayerRenderVideoNew(cstrName.c_str(),
        cstrFilePath.c_str(), render_type, offset_x, offset_y, zoom, repeat_enabled);
}

DslReturnType dsl_player_render_image_new(const wchar_t* name, const wchar_t* file_path,
    uint render_type, uint offset_x, uint offset_y, uint zoom, uint timeout)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(file_path);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFilePath(file_path);
    std::string cstrFilePath(wstrFilePath.begin(), wstrFilePath.end());

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

DslReturnType dsl_smtp_mail_enabled_get(boolean* enabled)
{
    return DSL::Services::GetServices()->SmtpMailEnabledGet(enabled);
}

DslReturnType dsl_smtp_mail_enabled_set(boolean enabled)
{
    return DSL::Services::GetServices()->SmtpMailEnabledSet(enabled);
}

DslReturnType dsl_smtp_credentials_set(const wchar_t* username, 
    const wchar_t* password)
{
    RETURN_IF_PARAM_IS_NULL(username);
    RETURN_IF_PARAM_IS_NULL(password);

    std::wstring wstrUsername(username);
    std::string cstrUsername(wstrUsername.begin(), wstrUsername.end());
    std::wstring wstrPassword(password);
    std::string cstrPassword(wstrPassword.begin(), wstrPassword.end());

    return DSL::Services::GetServices()->SmtpCredentialsSet(cstrUsername.c_str(), 
        cstrPassword.c_str());
}

DslReturnType dsl_smtp_server_url_get(const wchar_t** server_url)
{
    RETURN_IF_PARAM_IS_NULL(server_url);

    const char* cServerUrl;
    static std::string cstrServerUrl;
    static std::wstring wcstrServerUrl;
    
    DslReturnType result = DSL::Services::GetServices()->SmtpServerUrlGet(&cServerUrl);
    if (result == DSL_RESULT_SUCCESS)
    {
        cstrServerUrl.assign(cServerUrl);
        wcstrServerUrl.assign(cstrServerUrl.begin(), cstrServerUrl.end());
        *server_url = wcstrServerUrl.c_str();
    }
    return result;
}

DslReturnType dsl_smtp_server_url_set(const wchar_t* server_url)
{
    RETURN_IF_PARAM_IS_NULL(server_url);

    std::wstring wstrServerUrl(server_url);
    std::string cstrServerUrl(wstrServerUrl.begin(), wstrServerUrl.end());

    return DSL::Services::GetServices()->SmtpServerUrlSet(cstrServerUrl.c_str());
}

DslReturnType dsl_smtp_address_from_get(const wchar_t** name,
    const wchar_t** address)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(address);

    const char* cName;
    static std::string cstrName;
    static std::wstring wcstrName;
    const char* cAddress;
    static std::string cstrAddress;
    static std::wstring wcstrAddress;
    
    DslReturnType result = DSL::Services::GetServices()->SmtpFromAddressGet(&cName, &cAddress);
    if (result == DSL_RESULT_SUCCESS)
    {
        cstrName.assign(cName);
        wcstrName.assign(cstrName.begin(), cstrName.end());
        *name = wcstrName.c_str();
        cstrAddress.assign(cAddress);
        wcstrAddress.assign(cstrAddress.begin(), cstrAddress.end());
        *address = wcstrAddress.c_str();
    }
    return result;
}

DslReturnType dsl_smtp_address_from_set(const wchar_t* name,
    const wchar_t* address)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(address);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAddress(address);
    std::string cstrAddress(wstrAddress.begin(), wstrAddress.end());

    return DSL::Services::GetServices()->SmtpFromAddressSet(cstrName.c_str(),
        cstrAddress.c_str());
}    

DslReturnType dsl_smtp_ssl_enabled_get(boolean* enabled)
{
    return DSL::Services::GetServices()->SmtpSslEnabledGet(enabled);
}

DslReturnType dsl_smtp_ssl_enabled_set(boolean enabled)
{
    return DSL::Services::GetServices()->SmtpSslEnabledSet(enabled);
}

DslReturnType dsl_smtp_address_to_add(const wchar_t* name,
    const wchar_t* address)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(address);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAddress(address);
    std::string cstrAddress(wstrAddress.begin(), wstrAddress.end());

    return DSL::Services::GetServices()->SmtpToAddressAdd(cstrName.c_str(),
        cstrAddress.c_str());
    
}    
    
DslReturnType dsl_smtp_address_to_remove_all()
{
    return DSL::Services::GetServices()->SmtpToAddressesRemoveAll();
}
    
DslReturnType dsl_smtp_address_cc_add(const wchar_t* name,
    const wchar_t* address)
{
    RETURN_IF_PARAM_IS_NULL(name);
    RETURN_IF_PARAM_IS_NULL(address);

    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAddress(address);
    std::string cstrAddress(wstrAddress.begin(), wstrAddress.end());

    return DSL::Services::GetServices()->SmtpCcAddressAdd(cstrName.c_str(),
        cstrAddress.c_str());
}    

DslReturnType dsl_smtp_address_cc_remove_all()
{
    return DSL::Services::GetServices()->SmtpCcAddressesRemoveAll();
}

DslReturnType dsl_smtp_test_message_send()
{
     return DSL::Services::GetServices()->SendSmtpTestMessage();
}    
    
void dsl_delete_all()
{
    dsl_pipeline_delete_all();
    dsl_player_delete_all();
    dsl_component_delete_all();
    dsl_pph_delete_all();
    dsl_ode_trigger_delete_all();
    dsl_ode_area_delete_all();
    dsl_ode_action_delete_all();
    dsl_display_type_delete_all();
}


