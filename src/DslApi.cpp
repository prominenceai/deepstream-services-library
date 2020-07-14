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
#include "DslServices.h"

// Single GST debug catagory initialization
GST_DEBUG_CATEGORY(GST_CAT_DSL);

DslReturnType dsl_display_type_rgba_color_new(const wchar_t* name, 
    double red, double green, double blue, double alpha)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaColorNew(cstrName.c_str(), 
        red, green, blue, alpha);
}

DslReturnType dsl_display_type_rgba_font_new(const wchar_t* name, uint size, const wchar_t* color)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaFontNew(cstrName.c_str(), 
        size, cstrColor.c_str());
}

DslReturnType dsl_display_type_rgba_text_new(const wchar_t* name, const wchar_t* text, 
    uint x_offset, uint y_offset, const wchar_t* font, boolean has_bg_color, const wchar_t* bg_color)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrText(text);
    std::string cstrText(wstrText.begin(), wstrText.end());
    std::wstring wstrFont(font);
    std::string cstrFont(wstrFont.begin(), wstrFont.end());
    std::wstring wstrBgColor(bg_color);
    std::string cstrBgColor(wstrBgColor.begin(), wstrBgColor.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaTextNew(cstrName.c_str(), cstrText.c_str(),
        x_offset, y_offset, cstrFont.c_str(), has_bg_color, cstrBgColor.c_str());
}

DslReturnType dsl_display_type_rgba_line_new(const wchar_t* name, 
    uint x1, uint y1, uint x2, uint y2, uint width, const wchar_t* color)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaLineNew(cstrName.c_str(), 
        x1, y1, x2, y2, width, cstrColor.c_str());
}
    
DslReturnType dsl_display_type_rgba_rectangle_new(const wchar_t* name, uint left, uint top, uint width, uint height, 
    uint border_width, const wchar_t* color, bool has_bg_color, const wchar_t* bg_color)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());
    std::wstring wstrBgColor(bg_color);
    std::string cstrBgColor(wstrBgColor.begin(), wstrBgColor.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaRectangleNew(cstrName.c_str(), 
        left, top, width, height, border_width, cstrColor.c_str(), has_bg_color, cstrBgColor.c_str());
}
    
DslReturnType dsl_display_type_rgba_circle_new(const wchar_t* name, uint x_center, uint y_center, uint radius,
    const wchar_t* color, bool has_bg_color, const wchar_t* bg_color)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrColor(color);
    std::string cstrColor(wstrColor.begin(), wstrColor.end());
    std::wstring wstrBgColor(bg_color);
    std::string cstrBgColor(wstrBgColor.begin(), wstrBgColor.end());

    return DSL::Services::GetServices()->DisplayTypeRgbaCircleNew(cstrName.c_str(), 
        x_center, y_center, radius, cstrColor.c_str(), has_bg_color, cstrBgColor.c_str());
}
    
DslReturnType dsl_display_type_delete(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DisplayTypeDelete(cstrName.c_str());
}

DslReturnType dsl_display_type_delete_many(const wchar_t** names)
{
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

DslReturnType dsl_ode_action_callback_new(const wchar_t* name, 
    dsl_ode_handle_occurrence_cb client_hanlder, void* client_data)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionCallbackNew(cstrName.c_str(), client_hanlder, client_data);
}

DslReturnType dsl_ode_action_capture_frame_new(const wchar_t* name, const wchar_t* outdir)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOutdir(outdir);
    std::string cstrOutdir(wstrOutdir.begin(), wstrOutdir.end());

    return DSL::Services::GetServices()->OdeActionCaptureFrameNew(cstrName.c_str(), cstrOutdir.c_str());
}

DslReturnType dsl_ode_action_capture_object_new(const wchar_t* name, const wchar_t* outdir)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOutdir(outdir);
    std::string cstrOutdir(wstrOutdir.begin(), wstrOutdir.end());

    return DSL::Services::GetServices()->OdeActionCaptureObjectNew(cstrName.c_str(), cstrOutdir.c_str());
}

DslReturnType dsl_ode_action_display_new(const wchar_t* name,
    uint offsetX, uint offsetY, boolean offsetY_with_classId)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionDisplayNew(cstrName.c_str(),
        offsetX, offsetY, offsetY_with_classId);
}

DslReturnType dsl_ode_action_handler_disable_new(const wchar_t* name, const wchar_t* handler)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrHandler(handler);
    std::string cstrHandler(wstrHandler.begin(), wstrHandler.end());

    return DSL::Services::GetServices()->OdeActionHandlerDisableNew(cstrName.c_str(), 
        cstrHandler.c_str());
}

DslReturnType dsl_ode_action_hide_new(const wchar_t* name, boolean text, boolean border)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionHideNew(cstrName.c_str(), text, border);
}

DslReturnType dsl_ode_action_fill_area_new(const wchar_t* name,
    const wchar_t* area, double red, double green, double blue, double alpha)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrArea(area);
    std::string cstrArea(wstrArea.begin(), wstrArea.end());

    return DSL::Services::GetServices()->OdeActionFillAreaNew(cstrName.c_str(),
        cstrArea.c_str(), red, green, blue, alpha);
}

DslReturnType dsl_ode_action_fill_frame_new(const wchar_t* name,
    double red, double green, double blue, double alpha)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionFillFrameNew(cstrName.c_str(),
        red, green, blue, alpha);
}

DslReturnType dsl_ode_action_fill_object_new(const wchar_t* name,
    double red, double green, double blue, double alpha)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionFillObjectNew(cstrName.c_str(),
        red, green, blue, alpha);
}

DslReturnType dsl_ode_action_log_new(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionLogNew(cstrName.c_str());
}

DslReturnType dsl_ode_action_overlay_frame_new(const wchar_t* name, const wchar_t* display_type)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrType(display_type);
    std::string cstrType(wstrType.begin(), wstrType.end());

    return DSL::Services::GetServices()->OdeActionOverlayFrameNew(cstrName.c_str(), 
        cstrType.c_str());
}

DslReturnType dsl_ode_action_pause_new(const wchar_t* name, const wchar_t* pipeline)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->OdeActionPauseNew(cstrName.c_str(), 
        cstrPipeline.c_str());
}

DslReturnType dsl_ode_action_print_new(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionPrintNew(cstrName.c_str());
}

DslReturnType dsl_ode_action_sink_record_start_new(const wchar_t* name,
    const wchar_t* record_sink, uint start, uint duration, void* client_data)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrSink(record_sink);
    std::string cstrSink(wstrSink.begin(), wstrSink.end());

    return DSL::Services::GetServices()->OdeActionRecordStartNew(cstrName.c_str(), 
        cstrSink.c_str(), start, duration, client_data);
}

DslReturnType dsl_ode_action_redact_new(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionRedactNew(cstrName.c_str());
}

DslReturnType dsl_ode_action_sink_add_new(const wchar_t* name,
    const wchar_t* pipeline, const wchar_t* sink)
{
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
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrSink(sink);
    std::string cstrSink(wstrSink.begin(), wstrSink.end());
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->OdeActionSinkRemoveNew(cstrName.c_str(),
        cstrPipeline.c_str(), cstrSink.c_str());
}

DslReturnType dsl_ode_action_source_add_new(const wchar_t* name,
    const wchar_t* pipeline, const wchar_t* source)
{
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
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrSource(source);
    std::string cstrSource(wstrSource.begin(), wstrSource.end());
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->OdeActionSourceRemoveNew(cstrName.c_str(),
        cstrPipeline.c_str(), cstrSource.c_str());
}

DslReturnType dsl_ode_action_area_add_new(const wchar_t* name,
    const wchar_t* trigger, const wchar_t* area)
{
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
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrTrigger(trigger);
    std::string cstrTrigger(wstrTrigger.begin(), wstrTrigger.end());

    return DSL::Services::GetServices()->OdeActionTriggerResetNew(cstrName.c_str(),
        cstrTrigger.c_str());
}

DslReturnType dsl_ode_action_trigger_disable_new(const wchar_t* name, const wchar_t* trigger)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrTrigger(trigger);
    std::string cstrTrigger(wstrTrigger.begin(), wstrTrigger.end());

    return DSL::Services::GetServices()->OdeActionTriggerDisableNew(cstrName.c_str(),
        cstrTrigger.c_str());
}

DslReturnType dsl_ode_action_trigger_enable_new(const wchar_t* name, const wchar_t* trigger)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrTrigger(trigger);
    std::string cstrTrigger(wstrTrigger.begin(), wstrTrigger.end());

    return DSL::Services::GetServices()->OdeActionTriggerEnableNew(cstrName.c_str(),
        cstrTrigger.c_str());
}

DslReturnType dsl_ode_action_action_disable_new(const wchar_t* name, const wchar_t* action)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAction(action);
    std::string cstrAction(wstrAction.begin(), wstrAction.end());

    return DSL::Services::GetServices()->OdeActionActionDisableNew(cstrName.c_str(),
        cstrAction.c_str());
}

DslReturnType dsl_ode_action_action_enable_new(const wchar_t* name, const wchar_t* action)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAction(action);
    std::string cstrAction(wstrAction.begin(), wstrAction.end());

    return DSL::Services::GetServices()->OdeActionActionEnableNew(cstrName.c_str(),
        cstrAction.c_str());
}

DslReturnType dsl_ode_action_enabled_get(const wchar_t* name, boolean* enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionEnabledGet(cstrName.c_str(), enabled);
}

DslReturnType dsl_ode_action_enabled_set(const wchar_t* name, boolean enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionEnabledSet(cstrName.c_str(), enabled);
}

DslReturnType dsl_ode_action_delete(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeActionDelete(cstrName.c_str());
}

DslReturnType dsl_ode_action_delete_many(const wchar_t** names)
{
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

DslReturnType dsl_ode_area_new(const wchar_t* name, 
    uint left, uint top, uint width, uint height, boolean display)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeAreaNew(cstrName.c_str(), 
        left, top, width, height, display);
}

DslReturnType dsl_ode_area_get(const wchar_t* name, 
    uint* left, uint* top, uint* width, uint* height, boolean* display)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeAreaGet(cstrName.c_str(), 
        left, top, width, height, display);
}
    
DslReturnType dsl_ode_area_set(const wchar_t* name, 
    uint left, uint top, uint width, uint height, boolean display)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeAreaSet(cstrName.c_str(), 
        left, top, width, height, display);
}

DslReturnType dsl_ode_area_color_get(const wchar_t* name, 
    double* red, double* green, double* blue, double* alpha)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeAreaColorGet(cstrName.c_str(), 
        red, green, blue, alpha);
}
    
DslReturnType dsl_ode_area_color_set(const wchar_t* name, 
    double red, double green, double blue, double alpha)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeAreaColorSet(cstrName.c_str(), 
        red, green, blue, alpha);
}

DslReturnType dsl_ode_area_delete(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeAreaDelete(cstrName.c_str());
}

DslReturnType dsl_ode_area_delete_many(const wchar_t** names)
{
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

DslReturnType dsl_ode_trigger_always_new(const wchar_t* name, uint when)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerAlwaysNew(cstrName.c_str(), when);
}

DslReturnType dsl_ode_trigger_occurrence_new(const wchar_t* name, uint class_id, uint limit)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerOccurrenceNew(cstrName.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_absence_new(const wchar_t* name, uint class_id, uint limit)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerAbsenceNew(cstrName.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_intersection_new(const wchar_t* name, uint class_id, uint limit)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerIntersectionNew(cstrName.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_summation_new(const wchar_t* name, uint class_id, uint limit)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerSummationNew(cstrName.c_str(), class_id, limit);
}

DslReturnType dsl_ode_trigger_custom_new(const wchar_t* name, 
    uint class_id, uint limit, dsl_ode_check_for_occurrence_cb client_checker, void* client_data)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerCustomNew(cstrName.c_str(), 
        class_id, limit, client_checker, client_data);
}
    
DslReturnType dsl_ode_trigger_minimum_new(const wchar_t* name, 
    uint class_id, uint limit, uint minimum)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerMinimumNew(cstrName.c_str(), 
        class_id, limit, minimum);
}

DslReturnType dsl_ode_trigger_maximum_new(const wchar_t* name, 
    uint class_id, uint limit, uint maximum)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerMaximumNew(cstrName.c_str(), 
        class_id, limit, maximum);
}

DslReturnType dsl_ode_trigger_range_new(const wchar_t* name, 
    uint class_id, uint limit, uint lower, uint upper)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerRangeNew(cstrName.c_str(), 
        class_id, limit, lower, upper);
}

DslReturnType dsl_ode_trigger_reset(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerReset(cstrName.c_str());
}

DslReturnType dsl_ode_trigger_enabled_get(const wchar_t* name, boolean* enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerEnabledGet(cstrName.c_str(), enabled);
}

DslReturnType dsl_ode_trigger_enabled_set(const wchar_t* name, boolean enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerEnabledSet(cstrName.c_str(), enabled);
}

DslReturnType dsl_ode_trigger_class_id_get(const wchar_t* name, uint* class_id)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerClassIdGet(cstrName.c_str(), class_id);
}

DslReturnType dsl_ode_trigger_class_id_set(const wchar_t* name, uint class_id)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerClassIdSet(cstrName.c_str(), class_id);
}

DslReturnType dsl_ode_trigger_source_id_get(const wchar_t* name, uint* source_id)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerSourceIdGet(cstrName.c_str(), source_id);
}

DslReturnType dsl_ode_trigger_source_id_set(const wchar_t* name, uint source_id)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerSourceIdSet(cstrName.c_str(), source_id);
}

DslReturnType dsl_ode_trigger_confidence_min_get(const wchar_t* name, double* min_confidence)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerConfidenceMinGet(cstrName.c_str(), min_confidence);
}

DslReturnType dsl_ode_trigger_confidence_min_set(const wchar_t* name, double min_confidence)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerConfidenceMinSet(cstrName.c_str(), min_confidence);
}

DslReturnType dsl_ode_trigger_dimensions_min_get(const wchar_t* name, uint* min_width, uint* min_height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDimensionsMinGet(cstrName.c_str(), min_width, min_height);
}

DslReturnType dsl_ode_trigger_dimensions_min_set(const wchar_t* name, uint min_width, uint min_height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDimensionsMinSet(cstrName.c_str(), min_width, min_height);
}

DslReturnType dsl_ode_trigger_dimensions_max_get(const wchar_t* name, uint* max_width, uint* max_height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDimensionsMaxGet(cstrName.c_str(), max_width, max_height);
}

DslReturnType dsl_ode_trigger_dimensions_max_set(const wchar_t* name, uint max_width, uint max_height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDimensionsMaxSet(cstrName.c_str(), max_width, max_height);
}

DslReturnType dsl_ode_trigger_infer_done_only_get(const wchar_t* name, boolean* infer_done_only)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerInferDoneOnlyGet(cstrName.c_str(), infer_done_only);
}

DslReturnType dsl_ode_trigger_infer_done_only_set(const wchar_t* name, boolean infer_done_only)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerInferDoneOnlySet(cstrName.c_str(), infer_done_only);
}

DslReturnType dsl_ode_trigger_frame_count_min_get(const wchar_t* name, uint* min_count_n, uint* min_count_d)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerFrameCountMinGet(cstrName.c_str(), min_count_n, min_count_d);
}

DslReturnType dsl_ode_trigger_frame_count_min_set(const wchar_t* name, uint min_count_n, uint min_count_d)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerFrameCountMinSet(cstrName.c_str(), min_count_n, min_count_d);
}

DslReturnType dsl_ode_trigger_action_add(const wchar_t* name, const wchar_t* action)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAction(action);
    std::string cstrAction(wstrAction.begin(), wstrAction.end());

    return DSL::Services::GetServices()->OdeTriggerActionAdd(cstrName.c_str(), cstrAction.c_str());
}

DslReturnType dsl_ode_trigger_action_add_many(const wchar_t* name, const wchar_t** actions)
{
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
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAction(action);
    std::string cstrAction(wstrAction.begin(), wstrAction.end());

    return DSL::Services::GetServices()->OdeTriggerActionRemove(cstrName.c_str(), cstrAction.c_str());
}

DslReturnType dsl_ode_trigger_action_remove_many(const wchar_t* name, const wchar_t** actions)
{
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
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->OdeTriggerActionRemoveAll(cstrName.c_str());
}

DslReturnType dsl_ode_trigger_area_add(const wchar_t* name, const wchar_t* area)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrArea(area);
    std::string cstrArea(wstrArea.begin(), wstrArea.end());

    return DSL::Services::GetServices()->OdeTriggerAreaAdd(cstrName.c_str(), cstrArea.c_str());
}

DslReturnType dsl_ode_trigger_area_add_many(const wchar_t* name, const wchar_t** areas)
{
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
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrArea(area);
    std::string cstrArea(wstrArea.begin(), wstrArea.end());

    return DSL::Services::GetServices()->OdeTriggerAreaRemove(cstrName.c_str(), cstrArea.c_str());
}

DslReturnType dsl_trigger_ode_area_remove_many(const wchar_t* name, const wchar_t** areas)
{
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
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->OdeTriggerAreaRemoveAll(cstrName.c_str());
}

DslReturnType dsl_ode_trigger_delete(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeTriggerDelete(cstrName.c_str());
}

DslReturnType dsl_ode_trigger_delete_many(const wchar_t** names)
{
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

DslReturnType dsl_source_csi_new(const wchar_t* name, 
    uint width, uint height, uint fps_n, uint fps_d)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceCsiNew(cstrName.c_str(), 
        width, height, fps_n, fps_d);
}

DslReturnType dsl_source_usb_new(const wchar_t* name, 
    uint width, uint height, uint fps_n, uint fps_d)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceUsbNew(cstrName.c_str(), 
        width, height, fps_n, fps_d);
}

DslReturnType dsl_source_uri_new(const wchar_t* name, const wchar_t* uri, 
    boolean is_live, uint cudadec_mem_type, uint intra_decode, uint dropFrameInterval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUri(uri);
    std::string cstrUri(wstrUri.begin(), wstrUri.end());

    return DSL::Services::GetServices()->SourceUriNew(cstrName.c_str(), cstrUri.c_str(), 
        is_live, cudadec_mem_type, intra_decode, dropFrameInterval);
}

DslReturnType dsl_source_rtsp_new(const wchar_t* name, const wchar_t* uri,
    uint protocol, uint cudadec_mem_type, uint intra_decode, uint dropFrameInterval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUri(uri);
    std::string cstrUri(wstrUri.begin(), wstrUri.end());

    return DSL::Services::GetServices()->SourceRtspNew(cstrName.c_str(), cstrUri.c_str(), 
        protocol, cudadec_mem_type, intra_decode, dropFrameInterval);
}

DslReturnType dsl_source_dimensions_get(const wchar_t* name, uint* width, uint* height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceDimensionsGet(cstrName.c_str(), width, height);
}

DslReturnType dsl_source_frame_rate_get(const wchar_t* name, uint* fps_n, uint* fps_d)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceFrameRateGet(cstrName.c_str(), fps_n, fps_d);
}

DslReturnType dsl_source_decode_uri_get(const wchar_t* name, const wchar_t** uri)
{
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
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrUri(uri);
    std::string cstrUri(wstrUri.begin(), wstrUri.end());

    return DSL::Services::GetServices()->SourceDecodeUriSet(cstrName.c_str(), cstrUri.c_str());
}

DslReturnType dsl_source_decode_dewarper_add(const wchar_t* name, const wchar_t* dewarper)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrDewarper(dewarper);
    std::string cstrDewarper(wstrDewarper.begin(), wstrDewarper.end());

    return DSL::Services::GetServices()->SourceDecodeDewarperAdd(cstrName.c_str(), cstrDewarper.c_str());
}

DslReturnType dsl_source_decode_dewarper_remove(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SourceDecodeDewarperRemove(cstrName.c_str());
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
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());

    return DSL::Services::GetServices()->DewarperNew(cstrName.c_str(), cstrConfig.c_str());
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

DslReturnType dsl_gie_primary_kitti_output_enabled_set(const wchar_t* name, boolean enabled, const wchar_t* file)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFile(file);
    std::string cstrFile(wstrFile.begin(), wstrFile.end());

    return DSL::Services::GetServices()->PrimaryGieKittiOutputEnabledSet(cstrName.c_str(), enabled, cstrFile.c_str());
}

DslReturnType dsl_gie_primary_batch_meta_handler_add(const wchar_t* name, uint pad, 
    dsl_batch_meta_handler_cb handler, void* user_data)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->PrimaryGieBatchMetaHandlerAdd(cstrName.c_str(), pad, handler, user_data);
}

DslReturnType dsl_gie_primary_batch_meta_handler_remove(const wchar_t* name, uint pad,
    dsl_batch_meta_handler_cb handler)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->PrimaryGieBatchMetaHandlerRemove(cstrName.c_str(), pad, handler);
}

DslReturnType dsl_gie_secondary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* model_engine_file, const wchar_t* infer_on_gie, uint interval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(infer_config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());
    std::wstring wstrEngine(model_engine_file);
    std::string cstrEngine(wstrEngine.begin(), wstrEngine.end());
    std::wstring wstrInferOnGie(infer_on_gie);
    std::string cstrInferOnGie(wstrInferOnGie.begin(), wstrInferOnGie.end());
    
    return DSL::Services::GetServices()->SecondaryGieNew(cstrName.c_str(), cstrConfig.c_str(),
        cstrEngine.c_str(), cstrInferOnGie.c_str(), interval);
}

DslReturnType dsl_gie_infer_config_file_get(const wchar_t* name, const wchar_t** infer_config_file)
{
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
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(infer_config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());

    return DSL::Services::GetServices()->GieInferConfigFileSet(cstrName.c_str(), cstrConfig.c_str());
}

DslReturnType dsl_gie_model_engine_file_get(const wchar_t* name, const wchar_t** model_engine_file)
{
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
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrEngine(model_engine_file);
    std::string cstrEngine(wstrEngine.begin(), wstrEngine.end());

    return DSL::Services::GetServices()->GieModelEngineFileSet(cstrName.c_str(), cstrEngine.c_str());
}

DslReturnType dsl_gie_interval_get(const wchar_t* name, uint* interval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->GieIntervalGet(cstrName.c_str(), interval);
}

DslReturnType dsl_gie_interval_set(const wchar_t* name, uint interval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->GieIntervalSet(cstrName.c_str(), interval);
}


DslReturnType dsl_gie_raw_output_enabled_set(const wchar_t* name, boolean enabled, const wchar_t* path)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPath(path);
    std::string cstrPath(wstrPath.begin(), wstrPath.end());

    return DSL::Services::GetServices()->GieRawOutputEnabledSet(cstrName.c_str(), enabled, cstrPath.c_str());
}

DslReturnType dsl_tracker_ktl_new(const wchar_t* name, uint width, uint height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TrackerKtlNew(cstrName.c_str(), width, height);
}
    
DslReturnType dsl_tracker_iou_new(const wchar_t* name, const wchar_t* config_file, uint width, uint height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFile(config_file);
    std::string cstrFile(wstrFile.begin(), wstrFile.end());

    return DSL::Services::GetServices()->TrackerIouNew(cstrName.c_str(), cstrFile.c_str(), width, height);
}

DslReturnType dsl_tracker_max_dimensions_get(const wchar_t* name, uint* width, uint* height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TrackerMaxDimensionsGet(cstrName.c_str(), width, height);
}

DslReturnType dsl_tracker_max_dimensions_set(const wchar_t* name, uint width, uint height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TrackerMaxDimensionsSet(cstrName.c_str(), width, height);
}

DslReturnType dsl_tracker_batch_meta_handler_add(const wchar_t* name, uint pad, 
    dsl_batch_meta_handler_cb handler, void* user_data)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->TrackerBatchMetaHandlerAdd(cstrName.c_str(), pad, handler, user_data);
}

DslReturnType dsl_tracker_batch_meta_handler_remove(const wchar_t* name, uint pad,
    dsl_batch_meta_handler_cb handler)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->TrackerBatchMetaHandlerRemove(cstrName.c_str(), pad, handler);
}

DslReturnType dsl_tracker_kitti_output_enabled_set(const wchar_t* name, boolean enabled, const wchar_t* file)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFile(file);
    std::string cstrFile(wstrFile.begin(), wstrFile.end());

    return DSL::Services::GetServices()->TrackerKittiOutputEnabledSet(cstrName.c_str(), enabled, cstrFile.c_str());
}
    
DslReturnType dsl_ode_handler_new(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeHandlerNew(cstrName.c_str());
}

DslReturnType dsl_ode_handler_enabled_get(const wchar_t* name, boolean* enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeHandlerEnabledGet(cstrName.c_str(), enabled);
}

DslReturnType dsl_ode_handler_enabled_set(const wchar_t* name, boolean enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OdeHandlerEnabledSet(cstrName.c_str(), enabled);
}

DslReturnType dsl_ode_handler_trigger_add(const wchar_t* handler, const wchar_t* trigger)
{
    std::wstring wstrOdeHandler(handler);
    std::string cstrOdeHandler(wstrOdeHandler.begin(), wstrOdeHandler.end());
    std::wstring wstrOdeTrigger(trigger);
    std::string cstrOdeTrigger(wstrOdeTrigger.begin(), wstrOdeTrigger.end());

    return DSL::Services::GetServices()->OdeHandlerTriggerAdd(cstrOdeHandler.c_str(), cstrOdeTrigger.c_str());
}

DslReturnType dsl_ode_handler_trigger_add_many(const wchar_t* handler, const wchar_t** triggers)
{
    std::wstring wstrOdeHandler(handler);
    std::string cstrOdeHandler(wstrOdeHandler.begin(), wstrOdeHandler.end());

    for (const wchar_t** trigger = triggers; *trigger; trigger++)
    {
        std::wstring wstrOdeTrigger(*trigger);
        std::string cstrOdeTrigger(wstrOdeTrigger.begin(), wstrOdeTrigger.end());
        DslReturnType retval = DSL::Services::GetServices()->
            OdeHandlerTriggerAdd(cstrOdeHandler.c_str(), cstrOdeTrigger.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_ode_handler_trigger_remove(const wchar_t* handler, const wchar_t* trigger)
{
    std::wstring wstrOdeHandler(handler);
    std::string cstrOdeHandler(wstrOdeHandler.begin(), wstrOdeHandler.end());
    std::wstring wstrOdeTrigger(trigger);
    std::string cstrOdeTrigger(wstrOdeTrigger.begin(), wstrOdeTrigger.end());

    return DSL::Services::GetServices()->OdeHandlerTriggerRemove(cstrOdeHandler.c_str(), cstrOdeTrigger.c_str());
}

DslReturnType dsl_ode_handler_trigger_remove_many(const wchar_t* handler, const wchar_t** triggers)
{
    std::wstring wstrOdeHandler(handler);
    std::string cstrOdeHandler(wstrOdeHandler.begin(), wstrOdeHandler.end());

    for (const wchar_t** trigger = triggers; *trigger; trigger++)
    {
        std::wstring wstrOdeTrigger(*trigger);
        std::string cstrOdeTrigger(wstrOdeTrigger.begin(), wstrOdeTrigger.end());
        DslReturnType retval = DSL::Services::GetServices()->OdeHandlerTriggerRemove(cstrOdeHandler.c_str(), cstrOdeTrigger.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_ode_handler_trigger_remove_all(const wchar_t* handler)
{
    std::wstring wstrOdeHandler(handler);
    std::string cstrOdeHandler(wstrOdeHandler.begin(), wstrOdeHandler.end());

    return DSL::Services::GetServices()->OdeHandlerTriggerRemoveAll(cstrOdeHandler.c_str());
}

DslReturnType dsl_ofv_new(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OfvNew(cstrName.c_str());
}

DslReturnType dsl_osd_new(const wchar_t* name, boolean clock_enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdNew(cstrName.c_str(), clock_enabled);
}

DslReturnType dsl_osd_clock_enabled_get(const wchar_t* name, boolean* enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockEnabledGet(cstrName.c_str(), enabled);
}

DslReturnType dsl_osd_clock_enabled_set(const wchar_t* name, boolean enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockEnabledSet(cstrName.c_str(), enabled);
}

DslReturnType dsl_osd_clock_offsets_get(const wchar_t* name, uint* offsetX, uint* offsetY)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockOffsetsGet(cstrName.c_str(), offsetX, offsetY);
}

DslReturnType dsl_osd_clock_offsets_set(const wchar_t* name, uint offsetX, uint offsetY)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockOffsetsSet(cstrName.c_str(), offsetX, offsetY);
}

DslReturnType dsl_osd_clock_font_get(const wchar_t* name, const wchar_t** font, uint* size)
{
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
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFont(font);
    std::string cstrFont(wstrFont.begin(), wstrFont.end());

    return DSL::Services::GetServices()->OsdClockFontSet(cstrName.c_str(), cstrFont.c_str(), size);
}

DslReturnType dsl_osd_clock_color_get(const wchar_t* name, double* red, double* green, double* blue, double* alpha)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockColorGet(cstrName.c_str(), red, green, blue, alpha);
}

DslReturnType dsl_osd_clock_color_set(const wchar_t* name, double red, double green, double blue, double alpha)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockColorSet(cstrName.c_str(), red, green, blue, alpha);
}

DslReturnType dsl_osd_crop_settings_get(const wchar_t* name, 
    uint* left, uint* top, uint* width, uint* height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdCropSettingsGet(cstrName.c_str(), left, top, width, height);
}

DslReturnType dsl_osd_crop_settings_set(const wchar_t* name, 
    uint left, uint top, uint width, uint height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdCropSettingsSet(cstrName.c_str(), left, top, width, height);
}


DslReturnType dsl_osd_batch_meta_handler_add(const wchar_t* name, uint pad, 
    dsl_batch_meta_handler_cb handler, void* user_data)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->OsdBatchMetaHandlerAdd(cstrName.c_str(), pad, handler, user_data);
}

DslReturnType dsl_osd_batch_meta_handler_remove(const wchar_t* name, uint pad,
    dsl_batch_meta_handler_cb handler)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->OsdBatchMetaHandlerRemove(cstrName.c_str(), pad, handler);
}

DslReturnType dsl_tee_demuxer_new(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TeeDemuxerNew(cstrName.c_str());
}

DslReturnType dsl_tee_demuxer_new_branch_add_many(const wchar_t* tee, const wchar_t** branches)
{
    std::wstring wstrTee(tee);
    std::string cstrTee(wstrTee.begin(), wstrTee.end());

    DslReturnType retval = DSL::Services::GetServices()->TeeDemuxerNew(cstrTee.c_str());
    if (retval != DSL_RESULT_SUCCESS)
    {
        return retval;
    }

    for (const wchar_t** branch = branches; *branch; branch++)
    {
        std::wstring wstrBranch(*branch);
        std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        retval = DSL::Services::GetServices()->TeeBranchAdd(cstrTee.c_str(), cstrBranch.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_tee_splitter_new(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TeeSplitterNew(cstrName.c_str());
}

DslReturnType dsl_tee_splitter_new_branch_add_many(const wchar_t* tee, const wchar_t** branches)
{
    std::wstring wstrTee(tee);
    std::string cstrTee(wstrTee.begin(), wstrTee.end());

    DslReturnType retval = DSL::Services::GetServices()->TeeSplitterNew(cstrTee.c_str());
    if (retval != DSL_RESULT_SUCCESS)
    {
        return retval;
    }

    for (const wchar_t** branch = branches; *branch; branch++)
    {
        std::wstring wstrBranch(*branch);
        std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        retval = DSL::Services::GetServices()->TeeBranchAdd(cstrTee.c_str(), cstrBranch.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_tee_branch_add(const wchar_t* tee, const wchar_t* branch)
{
    std::wstring wstrTee(tee);
    std::string cstrTee(wstrTee.begin(), wstrTee.end());
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());

    return DSL::Services::GetServices()->TeeBranchAdd(cstrTee.c_str(), cstrBranch.c_str());
}

DslReturnType dsl_tee_branch_add_many(const wchar_t* tee, const wchar_t** branches)
{
    std::wstring wstrTee(tee);
    std::string cstrTee(wstrTee.begin(), wstrTee.end());
    
    for (const wchar_t** branch = branches; *branch; branch++)
    {
        std::wstring wstrBranch(*branch);
        std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        DslReturnType retval = DSL::Services::GetServices()->TeeBranchAdd(cstrTee.c_str(), cstrBranch.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_tee_branch_remove(const wchar_t* tee, const wchar_t* branch)
{
    std::wstring wstrTee(tee);
    std::string cstrTee(wstrTee.begin(), wstrTee.end());
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());

    return DSL::Services::GetServices()->TeeBranchRemove(cstrTee.c_str(), cstrBranch.c_str());
}

DslReturnType dsl_tee_branch_remove_many(const wchar_t* tee, const wchar_t** branches)
{
    std::wstring wstrTee(tee);
    std::string cstrTee(wstrTee.begin(), wstrTee.end());
    
    for (const wchar_t** branch = branches; *branch; branch++)
    {
        std::wstring wstrBranch(*branch);
        std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
        DslReturnType retval = DSL::Services::GetServices()->TeeBranchRemove(cstrTee.c_str(), cstrBranch.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_tee_branch_remove_all(const wchar_t* tee)
{
    std::wstring wstrTee(tee);
    std::string cstrTee(wstrTee.begin(), wstrTee.end());

    return DSL::Services::GetServices()->TeeBranchRemoveAll(cstrTee.c_str());
}

DslReturnType dsl_tee_branch_count_get(const wchar_t* tee, uint* count)
{
    std::wstring wstrTee(tee);
    std::string cstrTee(wstrTee.begin(), wstrTee.end());

    return DSL::Services::GetServices()->TeeBranchCountGet(cstrTee.c_str(), count);
}


DslReturnType dsl_tee_batch_meta_handler_add(const wchar_t* name,
    dsl_batch_meta_handler_cb handler, void* user_data)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->TeeBatchMetaHandlerAdd(cstrName.c_str(), handler, user_data);
}

DslReturnType dsl_tee_batch_meta_handler_remove(const wchar_t* name,
    dsl_batch_meta_handler_cb handler)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->TeeBatchMetaHandlerRemove(cstrName.c_str(), handler);
}

DslReturnType dsl_tiler_new(const wchar_t* name, uint width, uint height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TilerNew(cstrName.c_str(), width, height);
}

DslReturnType dsl_tiler_dimensions_get(const wchar_t* name, uint* width, uint* height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TilerDimensionsGet(cstrName.c_str(), width, height);
}

DslReturnType dsl_tiler_dimensions_set(const wchar_t* name, uint width, uint height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TilerDimensionsSet(cstrName.c_str(), width, height);
}

DslReturnType dsl_tiler_tiles_get(const wchar_t* name, uint* cols, uint* rows)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TilerTilesGet(cstrName.c_str(), cols, rows);
}

DslReturnType dsl_tiler_tiles_set(const wchar_t* name, uint cols, uint rows)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->TilerTilesSet(cstrName.c_str(), cols, rows);
}

DslReturnType dsl_tiler_batch_meta_handler_add(const wchar_t* name, uint pad, 
    dsl_batch_meta_handler_cb handler, void* user_data)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->TilerBatchMetaHandlerAdd(cstrName.c_str(), pad, handler, user_data);
}

DslReturnType dsl_tiler_batch_meta_handler_remove(const wchar_t* name, uint pad,
    dsl_batch_meta_handler_cb handler)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->TilerBatchMetaHandlerRemove(cstrName.c_str(), pad, handler);
}

DslReturnType dsl_sink_fake_new(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkFakeNew(cstrName.c_str());
}

DslReturnType dsl_sink_overlay_new(const wchar_t* name, uint overlay_id, uint display_id,
    uint depth, uint offsetX, uint offsetY, uint width, uint height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkOverlayNew(cstrName.c_str(), overlay_id, 
        display_id, depth, offsetX, offsetY, width, height);
}

DslReturnType dsl_sink_window_new(const wchar_t* name,
    uint offsetX, uint offsetY, uint width, uint height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkWindowNew(cstrName.c_str(), 
        offsetX, offsetY, width, height);
}

DslReturnType dsl_sink_file_new(const wchar_t* name, const wchar_t* filepath, 
     uint codec, uint container, uint bitrate, uint interval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrPath(filepath);
    std::string cstrPath(wstrPath.begin(), wstrPath.end());

    return DSL::Services::GetServices()->SinkFileNew(cstrName.c_str(), 
        cstrPath.c_str(), codec, container, bitrate, interval);
}     

DslReturnType dsl_sink_encode_video_formats_get(const wchar_t* name,
    uint* codec, uint* container)
{    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkEncodeVideoFormatsGet(cstrName.c_str(), 
        codec, container);
}

DslReturnType dsl_sink_encode_settings_get(const wchar_t* name,
    uint* bitrate, uint* interval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkEncodeSettingsGet(cstrName.c_str(), 
        bitrate, interval);
}    

DslReturnType dsl_sink_encode_settings_set(const wchar_t* name,
    uint bitrate, uint interval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkEncodeSettingsSet(cstrName.c_str(), 
        bitrate, interval);
}

DslReturnType dsl_sink_record_new(const wchar_t* name, const wchar_t* outdir, 
     uint codec, uint container, uint bitrate, uint interval, dsl_sink_record_client_listner_cb client_listener)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOutdir(outdir);
    std::string cstrOutdir(wstrOutdir.begin(), wstrOutdir.end());

    return DSL::Services::GetServices()->SinkRecordNew(cstrName.c_str(), 
        cstrOutdir.c_str(), codec, container, bitrate, interval, client_listener);
}     

DslReturnType dsl_sink_record_session_start(const wchar_t* name, 
     uint* session, uint start, uint duration,void* client_data)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordSessionStart(cstrName.c_str(), 
        session, start, duration, client_data);
}     

DslReturnType dsl_sink_record_session_stop(const wchar_t* name, uint session)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordSessionStop(cstrName.c_str(), session);
}

DslReturnType dsl_sink_record_cache_size_get(const wchar_t* name, uint* cache_size)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordCacheSizeGet(cstrName.c_str(), cache_size);
}

DslReturnType dsl_sink_record_cache_size_set(const wchar_t* name, uint cache_size)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordCacheSizeSet(cstrName.c_str(), cache_size);
}
 
DslReturnType dsl_sink_record_dimensions_get(const wchar_t* name, uint* width, uint* height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordDimensionsGet(cstrName.c_str(), width, height);
}

DslReturnType dsl_sink_record_dimensions_set(const wchar_t* name, uint width, uint height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordDimensionsSet(cstrName.c_str(), width, height);
}

DslReturnType dsl_sink_record_is_on_get(const wchar_t* name, boolean* is_on)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordIsOnGet(cstrName.c_str(), is_on);
}

DslReturnType dsl_sink_record_reset_done_get(const wchar_t* name, boolean* reset_done)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkRecordResetDoneGet(cstrName.c_str(), reset_done);
}

   
DslReturnType dsl_sink_rtsp_new(const wchar_t* name, const wchar_t* host, 
     uint udpPort, uint rtspPort, uint codec, uint bitrate, uint interval)
{
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
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkRtspServerSettingsGet(cstrName.c_str(), 
        udpPort, rtspPort, codec);
}    

DslReturnType dsl_sink_rtsp_encoder_settings_get(const wchar_t* name,
    uint* bitrate, uint* interval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkRtspEncoderSettingsGet(cstrName.c_str(), 
        bitrate, interval);
}    

DslReturnType dsl_sink_rtsp_encoder_settings_set(const wchar_t* name,
    uint bitrate, uint interval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkRtspEncoderSettingsSet(cstrName.c_str(), 
        bitrate, interval);
}

DslReturnType dsl_sink_image_new(const wchar_t* name, const wchar_t* outdir)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOutdir(outdir);
    std::string cstrOutdir(wstrOutdir.begin(), wstrOutdir.end());

    return DSL::Services::GetServices()->SinkImageNew(cstrName.c_str(), cstrOutdir.c_str());
}     

DslReturnType dsl_sink_image_outdir_get(const wchar_t* name, const wchar_t** outdir)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    const char* cOutdir;
    static std::string cstrOutdir;
    static std::wstring wcstrOutdir;
    
    uint retval = DSL::Services::GetServices()->SinkImageOutdirGet(cstrName.c_str(), &cOutdir);
    if (retval ==  DSL_RESULT_SUCCESS)
    {
        cstrOutdir.assign(cOutdir);
        wcstrOutdir.assign(cstrOutdir.begin(), cstrOutdir.end());
        *outdir = wcstrOutdir.c_str();
    }
    return retval;
}

DslReturnType dsl_sink_image_outdir_set(const wchar_t* name, const wchar_t* outdir)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrOutdir(outdir);
    std::string cstrOutdir(wstrOutdir.begin(), wstrOutdir.end());

    return DSL::Services::GetServices()->SinkImageOutdirSet(cstrName.c_str(), cstrOutdir.c_str());
}

DslReturnType dsl_sink_image_frame_capture_interval_get(const wchar_t* name, uint* interval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkImageFrameCaptureIntervalGet(cstrName.c_str(), interval);
}

DslReturnType dsl_sink_image_frame_capture_interval_set(const wchar_t* name, uint interval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkImageFrameCaptureIntervalSet(cstrName.c_str(), interval);
}

DslReturnType dsl_sink_image_frame_capture_enabled_get(const wchar_t* name, boolean* enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkImageFrameCaptureEnabledGet(cstrName.c_str(), enabled);
}

DslReturnType dsl_sink_image_frame_capture_enabled_set(const wchar_t* name, boolean enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkImageFrameCaptureEnabledSet(cstrName.c_str(), enabled);
}
    
DslReturnType dsl_sink_image_object_capture_enabled_get(const wchar_t* name, boolean* enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkImageObjectCaptureEnabledGet(cstrName.c_str(), enabled);
}

DslReturnType dsl_sink_image_object_capture_enabled_set(const wchar_t* name, boolean enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkImageObjectCaptureEnabledSet(cstrName.c_str(), enabled);
}

DslReturnType dsl_sink_image_object_capture_class_add(const wchar_t* name, uint classId, 
    boolean full_frame, uint capture_limit)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkImageObjectCaptureClassAdd(cstrName.c_str(), 
        classId, full_frame, capture_limit);
}

DslReturnType dsl_sink_image_object_capture_class_remove(const wchar_t* name, uint classId)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkImageObjectCaptureClassRemove(cstrName.c_str(), classId);
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

DslReturnType dsl_component_delete(const wchar_t* component)
{
    std::wstring wstrComponent(component);
    std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());

    return DSL::Services::GetServices()->ComponentDelete(cstrComponent.c_str());
}

DslReturnType dsl_component_delete_many(const wchar_t** components)
{
    for (const wchar_t** component = components; *component; component++)
    {
        std::wstring wstrComponent(*component);
        std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());
        DslReturnType retval = DSL::Services::GetServices()->ComponentDelete(cstrComponent.c_str());
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

DslReturnType dsl_component_gpuid_get(const wchar_t* component, uint* gpuid)
{
    std::wstring wstrComponent(component);
    std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());

    return DSL::Services::GetServices()->ComponentGpuIdGet(cstrComponent.c_str(), gpuid);
}

DslReturnType dsl_component_gpuid_set(const wchar_t* component, uint gpuid)
{
    std::wstring wstrComponent(component);
    std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());

    return DSL::Services::GetServices()->ComponentGpuIdSet(cstrComponent.c_str(), gpuid);
}

DslReturnType dsl_component_gpuid_set_many(const wchar_t** components, uint gpuid)
{
    for (const wchar_t** component = components; *component; component++)
    {
        std::wstring wstrComponent(*component);
        std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());
        DslReturnType retval = DSL::Services::GetServices()->ComponentGpuIdSet(cstrComponent.c_str(), gpuid);
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_branch_new(const wchar_t* branch)
{
    std::wstring wstrName(branch);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->BranchNew(cstrName.c_str());
}

DslReturnType dsl_branch_new_component_add_many(const wchar_t* branch, 
    const wchar_t** components)
{
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());

    DslReturnType retval = DSL::Services::GetServices()->BranchNew(cstrBranch.c_str());
    if (retval != DSL_RESULT_SUCCESS)
    {
        return retval;
    }

    for (const wchar_t** component = components; *component; component++)
    {
        std::wstring wstrComponent(*component);
        std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());
        retval = DSL::Services::GetServices()->BranchComponentAdd(cstrBranch.c_str(), cstrComponent.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_branch_new_many(const wchar_t** names)
{
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


DslReturnType dsl_branch_component_add(const wchar_t* branch, 
    const wchar_t* component)
{
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
    std::wstring wstrComponent(component);
    std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());

    return DSL::Services::GetServices()->BranchComponentAdd(cstrBranch.c_str(), cstrComponent.c_str());
}

DslReturnType dsl_branch_component_add_many(const wchar_t* branch, 
    const wchar_t** components)
{
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
    
    for (const wchar_t** component = components; *component; component++)
    {
        std::wstring wstrComponent(*component);
        std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());
        DslReturnType retval = DSL::Services::GetServices()->BranchComponentAdd(cstrBranch.c_str(), cstrComponent.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_branch_component_remove(const wchar_t* branch, 
    const wchar_t* component)
{
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
    std::wstring wstrComponent(component);
    std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());

    return DSL::Services::GetServices()->BranchComponentRemove(cstrBranch.c_str(), cstrComponent.c_str());
}

DslReturnType dsl_branch_component_remove_many(const wchar_t* branch, 
    const wchar_t** components)
{
    std::wstring wstrBranch(branch);
    std::string cstrBranch(wstrBranch.begin(), wstrBranch.end());
    
    for (const wchar_t** component = components; *component; component++)
    {
        std::wstring wstrComponent(*component);
        std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());
        DslReturnType retval = DSL::Services::GetServices()->BranchComponentRemove(cstrBranch.c_str(), cstrComponent.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_pipeline_new(const wchar_t* pipeline)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineNew(cstrPipeline.c_str());
}

DslReturnType dsl_pipeline_new_component_add_many(const wchar_t* pipeline, 
    const wchar_t** components)
{
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
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineDelete(cstrPipeline.c_str());
}

DslReturnType dsl_pipeline_delete_many(const wchar_t** pipelines)
{
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
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
    std::wstring wstrComponent(component);
    std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());

    return DSL::Services::GetServices()->PipelineComponentAdd(cstrPipeline.c_str(), cstrComponent.c_str());
}

DslReturnType dsl_pipeline_component_add_many(const wchar_t* pipeline, 
    const wchar_t** components)
{
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
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
    std::wstring wstrComponent(component);
    std::string cstrComponent(wstrComponent.begin(), wstrComponent.end());

    return DSL::Services::GetServices()->PipelineComponentRemove(cstrPipeline.c_str(), cstrComponent.c_str());
}

DslReturnType dsl_pipeline_component_remove_many(const wchar_t* pipeline, 
    const wchar_t** components)
{
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

DslReturnType dsl_pipeline_xwindow_clear(const wchar_t* pipeline)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineXWindowClear(cstrPipeline.c_str());
}
 
DslReturnType dsl_pipeline_xwindow_dimensions_get(const wchar_t* pipeline, 
    uint* width, uint* height)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineXWindowDimensionsGet(cstrPipeline.c_str(),
        width, height);
}

DslReturnType dsl_pipeline_xwindow_dimensions_set(const wchar_t* pipeline, 
    uint width, uint height)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineXWindowDimensionsSet(cstrPipeline.c_str(),
        width, height);
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

DslReturnType dsl_pipeline_state_get(const wchar_t* pipeline, uint* state)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineStateGet(cstrPipeline.c_str(), state);
}

DslReturnType dsl_pipeline_is_live(const wchar_t* pipeline, boolean* is_live)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineIsLive(cstrPipeline.c_str(), is_live);
}

DslReturnType dsl_pipeline_dump_to_dot(const wchar_t* pipeline, wchar_t* filename)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
    std::wstring wstrFilename(filename);
    std::string cstrFilename(wstrFilename.begin(), wstrFilename.end());

    return DSL::Services::GetServices()->PipelineDumpToDot(cstrPipeline.c_str(), 
        const_cast<char*>(cstrFilename.c_str()));
}

DslReturnType dsl_pipeline_dump_to_dot_with_ts(const wchar_t* pipeline, wchar_t* filename)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());
    std::wstring wstrFilename(filename);
    std::string cstrFilename(wstrFilename.begin(), wstrFilename.end());

    return DSL::Services::GetServices()->PipelineDumpToDotWithTs(cstrPipeline.c_str(), 
        const_cast<char*>(cstrFilename.c_str()));
}

DslReturnType dsl_pipeline_state_change_listener_add(const wchar_t* pipeline, 
    dsl_state_change_listener_cb listener, void* userdata)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineStateChangeListenerAdd(cstrPipeline.c_str(), listener, userdata);
}

DslReturnType dsl_pipeline_state_change_listener_remove(const wchar_t* pipeline, 
    dsl_state_change_listener_cb listener)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineStateChangeListenerRemove(cstrPipeline.c_str(), listener);
}

DslReturnType dsl_pipeline_eos_listener_add(const wchar_t* pipeline, 
    dsl_eos_listener_cb listener, void* userdata)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineEosListenerAdd(cstrPipeline.c_str(), listener, userdata);
}

DslReturnType dsl_pipeline_eos_listener_remove(const wchar_t* pipeline, 
    dsl_eos_listener_cb listener)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineEosListenerRemove(cstrPipeline.c_str(), listener);
}

DslReturnType dsl_pipeline_xwindow_key_event_handler_add(const wchar_t* pipeline, 
    dsl_xwindow_key_event_handler_cb handler, void* user_data)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineXWindowKeyEventHandlerAdd(cstrPipeline.c_str(), handler, user_data);
}    

DslReturnType dsl_pipeline_xwindow_key_event_handler_remove(const wchar_t* pipeline, 
    dsl_xwindow_key_event_handler_cb handler)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineXWindowKeyEventHandlerRemove(cstrPipeline.c_str(), handler);
}

DslReturnType dsl_pipeline_xwindow_button_event_handler_add(const wchar_t* pipeline, 
    dsl_xwindow_button_event_handler_cb handler, void* user_data)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineXWindowButtonEventHandlerAdd(cstrPipeline.c_str(), handler, user_data);
}    

DslReturnType dsl_pipeline_xwindow_button_event_handler_remove(const wchar_t* pipeline, 
    dsl_xwindow_button_event_handler_cb handler)    
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineXWindowButtonEventHandlerRemove(cstrPipeline.c_str(), handler);
}

DslReturnType dsl_pipeline_xwindow_delete_event_handler_add(const wchar_t* pipeline, 
    dsl_xwindow_delete_event_handler_cb handler, void* user_data)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineXWindowDeleteEventHandlerAdd(cstrPipeline.c_str(), handler, user_data);
}    

DslReturnType dsl_pipeline_xwindow_delete_event_handler_remove(const wchar_t* pipeline, 
    dsl_xwindow_delete_event_handler_cb handler)    
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->
        PipelineXWindowDeleteEventHandlerRemove(cstrPipeline.c_str(), handler);
}

void dsl_delete_all()
{
    dsl_pipeline_delete_all();
    dsl_component_delete_all();
    dsl_ode_trigger_delete_all();
    dsl_ode_area_delete_all();
    dsl_ode_action_delete_all();
}


