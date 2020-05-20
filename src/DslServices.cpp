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
#include "DslDetectionEvent.h"
#include "DslServices.h"
#include "DslSourceBintr.h"
#include "DslGieBintr.h"
#include "DslTrackerBintr.h"
#include "DslReporterBintr.h"
#include "DslTilerBintr.h"
#include "DslOsdBintr.h"
#include "DslSinkBintr.h"

// Single GST debug catagory initialization
GST_DEBUG_CATEGORY(GST_CAT_DSL);

DslReturnType dsl_event_action_display_new(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->EventActionDisplayNew(cstrName.c_str());
}

DslReturnType dsl_event_action_callback_new(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->EventActionCallbackNew(cstrName.c_str());
}

DslReturnType dsl_event_action_delete(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->EventActionDelete(cstrName.c_str());
}

DslReturnType dsl_event_action_delete_many(const wchar_t** names)
{
    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());

        DslReturnType retval = DSL::Services::GetServices()->EventActionDelete(cstrName.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_event_action_delete_all()
{
    return DSL::Services::GetServices()->EventActionDeleteAll();
}

uint dsl_event_action_list_size()
{
    return DSL::Services::GetServices()->EventActionListSize();
}

DslReturnType dsl_detection_event_new(const wchar_t* name, uint evtype, uint class_id)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DetectionEventNew(cstrName.c_str(), evtype, class_id);
}

DslReturnType dsl_detection_event_class_id_get(const wchar_t* name, uint* class_id)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DetectionEventClassIdGet(cstrName.c_str(), class_id);
}

DslReturnType dsl_detection_event_class_id_set(const wchar_t* name, uint class_id)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DetectionEventClassIdSet(cstrName.c_str(), class_id);
}

DslReturnType dsl_detection_event_dimensions_min_get(const wchar_t* name, uint* min_width, uint* min_height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DetectionEventDimensionsMinGet(cstrName.c_str(), min_width, min_height);
}

DslReturnType dsl_detection_event_dimensions_min_set(const wchar_t* name, uint min_width, uint min_height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DetectionEventDimensionsMinSet(cstrName.c_str(), min_width, min_height);
}

DslReturnType dsl_detection_event_frame_count_min_get(const wchar_t* name, uint* min_count_n, uint* min_count_d)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DetectionEventFrameCountMinGet(cstrName.c_str(), min_count_n, min_count_d);
}

DslReturnType dsl_detection_event_frame_count_min_set(const wchar_t* name, uint min_count_n, uint min_count_d)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->DetectionEventFrameCountMinSet(cstrName.c_str(), min_count_n, min_count_d);
}

DslReturnType dsl_detection_event_action_add(const wchar_t* name, const wchar_t* action)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAction(action);
    std::string cstrAction(wstrAction.begin(), wstrAction.end());

    return DSL::Services::GetServices()->DetectionEventActionAdd(cstrName.c_str(), cstrAction.c_str());
}

DslReturnType dsl_detection_event_action_add_many(const wchar_t* name, const wchar_t** actions)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    for (const wchar_t** action = actions; *action; action++)
    {
        std::wstring wstrAction(*action);
        std::string cstrAction(wstrAction.begin(), wstrAction.end());
        
        DslReturnType retval = DSL::Services::GetServices()->
            DetectionEventActionAdd(cstrName.c_str(), cstrAction.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_detection_event_action_remove(const wchar_t* name, const wchar_t* action)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrAction(action);
    std::string cstrAction(wstrAction.begin(), wstrAction.end());

    return DSL::Services::GetServices()->DetectionEventActionRemove(cstrName.c_str(), cstrAction.c_str());
}

DslReturnType dsl_detection_event_action_remove_many(const wchar_t* name, const wchar_t** actions)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    for (const wchar_t** action = actions; *action; action++)
    {
        std::wstring wstrAction(*action);
        std::string cstrAction(wstrAction.begin(), wstrAction.end());
        
        DslReturnType retval = DSL::Services::GetServices()->
            DetectionEventActionRemove(cstrName.c_str(), cstrAction.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_detection_event_action_remove_all(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->DetectionEventActionRemoveAll(cstrName.c_str());
}

DslReturnType dsl_event_delete(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->EventDelete(cstrName.c_str());
}

DslReturnType dsl_event_delete_many(const wchar_t** names)
{
    for (const wchar_t** name = names; *name; name++)
    {
        std::wstring wstrName(*name);
        std::string cstrName(wstrName.begin(), wstrName.end());
        DslReturnType retval = DSL::Services::GetServices()->EventDelete(cstrName.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_event_delete_all()
{
    return DSL::Services::GetServices()->EventDeleteAll();
}

uint dsl_event_list_size()
{
    return DSL::Services::GetServices()->EventListSize();
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
    
DslReturnType dsl_reporter_new(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->ReporterNew(cstrName.c_str());
}

DslReturnType dsl_reporter_enabled_get(const wchar_t* name, boolean* enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->ReporterEnabledGet(cstrName.c_str(), enabled);
}

DslReturnType dsl_reporter_enabled_set(const wchar_t* name, boolean enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->ReporterEnabledSet(cstrName.c_str(), enabled);
}

DslReturnType dsl_reporter_detection_event_add(const wchar_t* reporter, const wchar_t* detection_event)
{
    std::wstring wstrReporter(reporter);
    std::string cstrReporter(wstrReporter.begin(), wstrReporter.end());
    std::wstring wstrEvent(detection_event);
    std::string cstrEvent(wstrEvent.begin(), wstrEvent.end());

    return DSL::Services::GetServices()->ReporterDetectionEventAdd(cstrReporter.c_str(), cstrEvent.c_str());
}

DslReturnType dsl_reporter_detection_event_add_many(const wchar_t* reporter, const wchar_t** detection_events)
{
    std::wstring wstrReporter(reporter);
    std::string cstrReporter(wstrReporter.begin(), wstrReporter.end());

    for (const wchar_t** detection_event = detection_events; *detection_event; detection_event++)
    {
        std::wstring wstrEvent(*detection_event);
        std::string cstrEvent(wstrEvent.begin(), wstrEvent.end());
        DslReturnType retval = DSL::Services::GetServices()->
            ReporterDetectionEventAdd(cstrReporter.c_str(), cstrEvent.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_reporter_detection_event_remove(const wchar_t* reporter, const wchar_t* detection_event)
{
    std::wstring wstrReporter(reporter);
    std::string cstrReporter(wstrReporter.begin(), wstrReporter.end());
    std::wstring wstrEvent(detection_event);
    std::string cstrEvent(wstrEvent.begin(), wstrEvent.end());

    return DSL::Services::GetServices()->ReporterDetectionEventRemove(cstrReporter.c_str(), cstrEvent.c_str());
}

DslReturnType dsl_reporter_detection_event_remove_many(const wchar_t* reporter, const wchar_t** detection_events)
{
    std::wstring wstrReporter(reporter);
    std::string cstrReporter(wstrReporter.begin(), wstrReporter.end());

    for (const wchar_t** detection_event = detection_events; *detection_event; detection_event++)
    {
        std::wstring wstrEvent(*detection_event);
        std::string cstrEvent(wstrEvent.begin(), wstrEvent.end());
        DslReturnType retval = DSL::Services::GetServices()->ReporterDetectionEventRemove(cstrReporter.c_str(), cstrEvent.c_str());
        if (retval != DSL_RESULT_SUCCESS)
        {
            return retval;
        }
    }
    return DSL_RESULT_SUCCESS;
}

DslReturnType dsl_reporter_detection_event_remove_all(const wchar_t* reporter)
{
    std::wstring wstrReporter(reporter);
    std::string cstrReporter(wstrReporter.begin(), wstrReporter.end());

    return DSL::Services::GetServices()->ReporterDetectionEventRemoveAll(cstrReporter.c_str());
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

DslReturnType dsl_osd_redaction_enabled_get(const wchar_t* name, boolean* enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdRedactionEnabledGet(cstrName.c_str(), enabled);
}

DslReturnType dsl_osd_redaction_enabled_set(const wchar_t* name, boolean enabled)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdRedactionEnabledSet(cstrName.c_str(), enabled);
}

DslReturnType dsl_osd_redaction_class_add(const wchar_t* name, int classId, 
    double red, double green, double blue, double alpha)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdRedactionClassAdd(cstrName.c_str(), classId, 
        red, green, blue, alpha);
}

DslReturnType dsl_osd_redaction_class_remove(const wchar_t* name, int classId)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdRedactionClassRemove(cstrName.c_str(), classId);
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

DslReturnType dsl_osd_kitti_output_enabled_set(const wchar_t* name, boolean enabled, const wchar_t* file)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFile(file);
    std::string cstrFile(wstrFile.begin(), wstrFile.end());

    return DSL::Services::GetServices()->OsdKittiOutputEnabledSet(cstrName.c_str(), enabled, cstrFile.c_str());
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

DslReturnType dsl_sink_file_video_formats_get(const wchar_t* name,
    uint* codec, uint* container)
{    
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkFileVideoFormatsGet(cstrName.c_str(), 
        codec, container);
}

DslReturnType dsl_sink_file_encoder_settings_get(const wchar_t* name,
    uint* bitrate, uint* interval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkFileEncoderSettingsGet(cstrName.c_str(), 
        bitrate, interval);
}    

DslReturnType dsl_sink_file_encoder_settings_set(const wchar_t* name,
    uint bitrate, uint interval)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->SinkFileEncoderSettingsSet(cstrName.c_str(), 
        bitrate, interval);
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

#define RETURN_IF_ACTION_NAME_NOT_FOUND(actions, name) do \
{ \
    if (actions.find(name) == actions.end()) \
    { \
        LOG_ERROR("Event Action name '" << name << "' was not found"); \
        return DSL_RESULT_ACTION_NAME_NOT_FOUND; \
    } \
}while(0); 

#define RETURN_IF_EVENT_NAME_NOT_FOUND(events, name) do \
{ \
    if (events.find(name) == events.end()) \
    { \
        LOG_ERROR("Event name '" << name << "' was not found"); \
        return DSL_RESULT_EVENT_NAME_NOT_FOUND; \
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
    
    DslReturnType Services::EventActionDisplayNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure event name uniqueness 
        if (m_eventActions.find(name) != m_eventActions.end())
        {   
            LOG_ERROR("Event Action name '" << name << "' is not unique");
            return DSL_RESULT_ACTION_NAME_NOT_UNIQUE;
        }
        try
        {
            m_eventActions[name] = DSL_EVENT_ACTION_DISPLAY_NEW(name);
        }
        catch(...)
        {
            LOG_ERROR("New Display Event Action '" << name << "' threw exception on create");
            return DSL_RESULT_ACTION_THREW_EXCEPTION;
        }
        LOG_INFO("new Display Event Action '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::EventActionCallbackNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure event name uniqueness 
        if (m_eventActions.find(name) != m_eventActions.end())
        {   
            LOG_ERROR("Event Action name '" << name << "' is not unique");
            return DSL_RESULT_ACTION_NAME_NOT_UNIQUE;
        }
        try
        {
            m_eventActions[name] = DSL_EVENT_ACTION_CALLBACK_NEW(name);
        }
        catch(...)
        {
            LOG_ERROR("New CallBack Event Action '" << name << "' threw exception on create");
            return DSL_RESULT_ACTION_THREW_EXCEPTION;
        }
        LOG_INFO("new Callback Event Action '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::EventActionDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_ACTION_NAME_NOT_FOUND(m_eventActions, name);
        
        if (m_eventActions[name]->IsInUse())
        {
            LOG_INFO("Event Action'" << name << "' is in use");
            return DSL_RESULT_ACTION_IN_USE;
        }
        m_eventActions.erase(name);

        LOG_INFO("Event Action '" << name << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::EventActionDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        for (auto const& imap: m_eventActions)
        {
            // In the case of Delete all
            if (imap.second->IsInUse())
            {
                LOG_ERROR("Event Action '" << imap.second->GetName() << "' is currently in use");
                return DSL_RESULT_ACTION_IN_USE;
            }
        }
        m_eventActions.clear();

        LOG_INFO("All Events deleted successfully");

        return DSL_RESULT_SUCCESS;
    }

    uint Services::EventActionListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_eventActions.size();
    }
    
    
    DslReturnType Services::DetectionEventNew(const char* name, uint evtype, uint classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure event name uniqueness 
        if (m_detectionEvents.find(name) != m_detectionEvents.end())
        {   
            LOG_ERROR("Detection Event name '" << name << "' is not unique");
            return DSL_RESULT_EVENT_NAME_NOT_UNIQUE;
        }
        try
        {
            switch (evtype)
            {
            case DSL_EVENT_TYPE_FIRST_OCCURRENCE :
                m_detectionEvents[name] = DSL_EVENT_FIRST_OCCURRENCE_NEW(name, classId);
                break;
            default :
                LOG_ERROR("New CSI Source '" << name << "' threw exception on create");
                return DSL_RESULT_EVENT_TYPE_INVALID;
            }
        }
        catch(...)
        {
            LOG_ERROR("New CSI Source '" << name << "' threw exception on create");
            return DSL_RESULT_EVENT_THREW_EXCEPTION;
        }
        LOG_INFO("new Detection Event '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::DetectionEventClassIdGet(const char* name, uint* classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_EVENT_NAME_NOT_FOUND(m_detectionEvents, name);
            
            DSL_DETECTION_EVENT_PTR pEvent = 
                std::dynamic_pointer_cast<DetectionEvent>(m_detectionEvents[name]);
         
            *classId = pEvent->GetClassId();
        }
        catch(...)
        {
            LOG_ERROR("Event '" << name << "' threw exception getting class id");
            return DSL_RESULT_EVENT_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }                

    DslReturnType Services::DetectionEventClassIdSet(const char* name, uint classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_EVENT_NAME_NOT_FOUND(m_detectionEvents, name);
            
            DSL_DETECTION_EVENT_PTR pEvent = 
                std::dynamic_pointer_cast<DetectionEvent>(m_detectionEvents[name]);
         
            pEvent->SetClassId(classId);
        }
        catch(...)
        {
            LOG_ERROR("Event '" << name << "' threw exception getting class id");
            return DSL_RESULT_EVENT_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }                

    DslReturnType Services::DetectionEventDimensionsMinGet(const char* name, uint* min_width, uint* min_height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_EVENT_NAME_NOT_FOUND(m_detectionEvents, name);
            
            DSL_DETECTION_EVENT_PTR pEvent = 
                std::dynamic_pointer_cast<DetectionEvent>(m_detectionEvents[name]);
         
            pEvent->GetMinDimensions(min_width, min_height);
        }
        catch(...)
        {
            LOG_ERROR("Event '" << name << "' threw exception getting minimum dimensions");
            return DSL_RESULT_EVENT_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }                

    DslReturnType Services::DetectionEventDimensionsMinSet(const char* name, uint min_width, uint min_height)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_EVENT_NAME_NOT_FOUND(m_detectionEvents, name);
            
            DSL_DETECTION_EVENT_PTR pEvent = 
                std::dynamic_pointer_cast<DetectionEvent>(m_detectionEvents[name]);
         
            // TODO: validate the min values for in-range
            pEvent->SetMinDimensions(min_width, min_height);
        }
        catch(...)
        {
            LOG_ERROR("Event '" << name << "' threw exception setting minimum dimensions");
            return DSL_RESULT_EVENT_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }                

    DslReturnType Services:: DetectionEventFrameCountMinGet(const char* name, uint* min_count_n, uint* min_count_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_EVENT_NAME_NOT_FOUND(m_detectionEvents, name);
            
            DSL_DETECTION_EVENT_PTR pEvent = 
                std::dynamic_pointer_cast<DetectionEvent>(m_detectionEvents[name]);
         
            pEvent->GetMinFrameCount(min_count_n, min_count_d);
        }
        catch(...)
        {
            LOG_ERROR("Event '" << name << "' threw exception getting minimum frame count");
            return DSL_RESULT_EVENT_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }                

    DslReturnType Services:: DetectionEventFrameCountMinSet(const char* name, uint min_count_n, uint min_count_d)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_EVENT_NAME_NOT_FOUND(m_detectionEvents, name);
            
            DSL_DETECTION_EVENT_PTR pEvent = 
                std::dynamic_pointer_cast<DetectionEvent>(m_detectionEvents[name]);
         
            // TODO: validate the min values for in-range
            pEvent->SetMinFrameCount(min_count_n, min_count_d);
        }
        catch(...)
        {
            LOG_ERROR("Event '" << name << "' threw exception getting minimum frame count");
            return DSL_RESULT_EVENT_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }                

    DslReturnType Services::DetectionEventActionAdd(const char* name, const char* action)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_EVENT_NAME_NOT_FOUND(m_detectionEvents, name);
        RETURN_IF_ACTION_NAME_NOT_FOUND(m_eventActions, action);

        try
        {
            // Can't add Actions if they're In use by another Detection Event
            if (m_eventActions[action]->IsInUse())
            {
                LOG_ERROR("Unable to add Event Action '" << action 
                    << "' as it is currently in use");
                return DSL_RESULT_ACTION_IN_USE;
            }

            if (!m_detectionEvents[name]->AddChild(m_eventActions[action]))
            {
                LOG_ERROR("Detection Event '" << name
                    << "' failed to add Event Action '" << action << "'");
                return DSL_RESULT_EVENT_ACTION_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Detection Event '" << name
                << "' threw exception adding Event Action '" << action << "'");
            return DSL_RESULT_EVENT_THREW_EXCEPTION;
        }
        LOG_INFO("Event Action '" << action
            << "' was added to Detection Event '" << name << "' successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::DetectionEventActionRemove(const char* name, const char* action)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_EVENT_NAME_NOT_FOUND(m_detectionEvents, name);
        RETURN_IF_ACTION_NAME_NOT_FOUND(m_eventActions, action);

        try
        {
            if (!m_eventActions[action]->IsParent(m_detectionEvents[name]))
            {
                LOG_ERROR("Event Action'" << action << 
                    "' is not in use by Detection Event '" << name << "'");
                return DSL_RESULT_EVENT_ACTION_NOT_IN_USE;
            }

            if (!m_detectionEvents[name]->RemoveChild(m_eventActions[action]))
            {
                LOG_ERROR("Detection Event '" << name
                    << "' failed to remove Event Action '" << action << "'");
                return DSL_RESULT_EVENT_ACTION_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Detection Event '" << name
                << "' threw exception remove Event Action '" << action << "'");
            return DSL_RESULT_EVENT_THREW_EXCEPTION;
        }
        LOG_INFO("Event Action '" << action
            << "' was removed from Detection Event '" << name << "' successfully");

        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::DetectionEventActionRemoveAll(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_EVENT_NAME_NOT_FOUND(m_detectionEvents, name);

        try
        {
            m_detectionEvents[name]->RemoveAllChildren();
        }
        catch(...)
        {
            LOG_ERROR("Detection Event '" << name 
                << "' threw an exception removing All Events Actions");
            return DSL_RESULT_EVENT_THREW_EXCEPTION;
        }
        LOG_INFO("All Events Actions removed from Detection Event '" << name << "' successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::EventDelete(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_EVENT_NAME_NOT_FOUND(m_detectionEvents, name);
        
        if (m_detectionEvents[name]->IsInUse())
        {
            LOG_INFO("Event '" << name << "' is in use");
            return DSL_RESULT_EVENT_IN_USE;
        }
        m_detectionEvents.erase(name);

        LOG_INFO("Event '" << name << "' deleted successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::EventDeleteAll()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        for (auto const& imap: m_detectionEvents)
        {
            // In the case of Delete all
            if (imap.second->IsInUse())
            {
                LOG_ERROR("Event '" << imap.second->GetName() << "' is currently in use");
                return DSL_RESULT_EVENT_IN_USE;
            }
        }
        m_detectionEvents.clear();

        LOG_INFO("All Events deleted successfully");

        return DSL_RESULT_SUCCESS;
    }

    uint Services::EventListSize()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        return m_detectionEvents.size();
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
        LOG_INFO("new CSI Source '" << name << "' created successfully");

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
        LOG_INFO("new USB Source '" << name << "' created successfully");

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
        LOG_INFO("new URI Source '" << name << "' created successfully");

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
        LOG_INFO("new RTSP Source '" << name << "' created successfully");

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
        LOG_INFO("new Dewarper '" << name << "' created successfully");

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
        
        LOG_INFO("Model engine file: " << modelEngineFile);
        
        std::ifstream modelFile(modelEngineFile);
        if (!modelFile.good())
        {
            LOG_ERROR("Model Engine File not found");
            return DSL_RESULT_GIE_MODEL_FILE_NOT_FOUND;
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
        LOG_INFO("new Primary GIE '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }


    DslReturnType Services::PrimaryGieBatchMetaHandlerAdd(const char* name, uint pad, dsl_batch_meta_handler_cb handler, void* user_data)
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

            if (!pPrimaryGieBintr->AddBatchMetaHandler(pad, handler, user_data))
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
        LOG_INFO("new Secondary GIE '" << name << "' created successfully");

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
        dsl_batch_meta_handler_cb handler, void* user_data)
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

            if (!pTrackerBintr->AddBatchMetaHandler(pad, handler, user_data))
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
        dsl_batch_meta_handler_cb handler, void* user_data)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        
        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_TEE(m_components, name);

            DSL_MULTI_COMPONENTS_PTR pTeeBintr = 
                std::dynamic_pointer_cast<MultiComponentsBintr>(m_components[name]);

            if (!pTeeBintr->AddBatchMetaHandler(DSL_PAD_SINK, handler, user_data))
            {
                LOG_ERROR("Tee '" << name << "' failed to add Batch Meta Handler");
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

            if (m_components[name]->IsInUse())
            {
                LOG_ERROR("Unable to set Dimensions for Tiler '" << name 
                    << "' as it's currently in use");
                return DSL_RESULT_TILER_IS_IN_USE;
            }

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

            if (m_components[name]->IsInUse())
            {
                LOG_ERROR("Unable to set Tiles for Tiler '" << name 
                    << "' as it's currently in use");
                return DSL_RESULT_TILER_IS_IN_USE;
            }

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
        dsl_batch_meta_handler_cb handler, void* user_data)
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

            if (!pTilerBintr->AddBatchMetaHandler(pad, handler, user_data))
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
   
    DslReturnType Services::ReporterNew(const char* name)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        // ensure component name uniqueness 
        if (m_components.find(name) != m_components.end())
        {   
            LOG_ERROR("Reporter name '" << name << "' is not unique");
            return DSL_RESULT_REPORTER_NAME_NOT_UNIQUE;
        }
        try
        {   
            m_components[name] = std::shared_ptr<Bintr>(new ReporterBintr(name));
        }
        catch(...)
        {
            LOG_ERROR("New Reporter '" << name << "' threw exception on create");
            return DSL_RESULT_REPORTER_THREW_EXCEPTION;
        }
        LOG_INFO("new Reporter '" << name << "' created successfully");

        return DSL_RESULT_SUCCESS;
    }
    
   DslReturnType Services::ReporterEnabledGet(const char* reporter, boolean* enabled)
   {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, reporter);

        try
        {
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, reporter, ReporterBintr);

            DSL_REPORTER_PTR pReporterBintr = 
                std::dynamic_pointer_cast<ReporterBintr>(m_components[reporter]);

            *enabled = pReporterBintr->GetReportingEnabled();
        }
        catch(...)
        {
            LOG_ERROR("Reporter '" << reporter
                << "' threw exception getting the Enabled state");
            return DSL_RESULT_REPORTER_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

   DslReturnType Services::ReporterEnabledSet(const char* reporter, boolean enabled)
   {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, reporter);

        try
        {
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, reporter, ReporterBintr);

            DSL_REPORTER_PTR pReporterBintr = 
                std::dynamic_pointer_cast<ReporterBintr>(m_components[reporter]);

            if (!pReporterBintr->SetReportingEnabled(enabled))
            {
                LOG_ERROR("Reporter '" << reporter
                    << "' failed to set enabled state");
                return DSL_RESULT_REPORTER_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Reporter '" << reporter
                << "' threw exception setting the Enabled state");
            return DSL_RESULT_REPORTER_THREW_EXCEPTION;
        }

        return DSL_RESULT_SUCCESS;
    }

   DslReturnType Services::ReporterDetectionEventAdd(const char* reporter, const char* detectionEvent)
   {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, reporter);
        RETURN_IF_EVENT_NAME_NOT_FOUND(m_detectionEvents, detectionEvent);

        try
        {
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, reporter, ReporterBintr);

            // Can't add Events if they're In use by another Reporter
            if (m_detectionEvents[detectionEvent]->IsInUse())
            {
                LOG_ERROR("Unable to add Detection Event '" << detectionEvent 
                    << "' as it is currently in use");
                return DSL_RESULT_EVENT_IN_USE;
            }

            DSL_REPORTER_PTR pReporterBintr = 
                std::dynamic_pointer_cast<ReporterBintr>(m_components[reporter]);

            if (!pReporterBintr->AddChild(m_detectionEvents[detectionEvent]))
            {
                LOG_ERROR("Reporter '" << reporter
                    << "' failed to add Detection Event '" << detectionEvent << "'");
                return DSL_RESULT_REPORTER_EVENT_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Reporter '" << reporter
                << "' threw exception adding Detection Event '" << detectionEvent << "'");
            return DSL_RESULT_REPORTER_THREW_EXCEPTION;
        }
        LOG_INFO("Detection Event '" << detectionEvent 
            << "' was added to Reporter '" << reporter << "' successfully");

        return DSL_RESULT_SUCCESS;
    }


    DslReturnType Services::ReporterDetectionEventRemove(const char* reporter, const char* detectionEvent)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, reporter);
        RETURN_IF_EVENT_NAME_NOT_FOUND(m_detectionEvents, detectionEvent);

        try
        {
            if (!m_detectionEvents[detectionEvent]->IsParent(m_components[reporter]))
            {
                LOG_ERROR("Detection Event '" << detectionEvent << 
                    "' is not in use by Reporter '" << reporter << "'");
                return DSL_RESULT_REPORTER_EVENT_NOT_IN_USE;
            }
            
            DSL_REPORTER_PTR pReporterBintr = 
                std::dynamic_pointer_cast<ReporterBintr>(m_components[reporter]);
                
            if (!pReporterBintr->RemoveChild(m_detectionEvents[detectionEvent]))
            {
                LOG_ERROR("Reporter '" << reporter
                    << "' failed to remove Detection Event '" << detectionEvent << "'");
                return DSL_RESULT_REPORTER_EVENT_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Reporter '" << reporter 
                << "' threw an exception removing Detection Event");
            return DSL_RESULT_REPORTER_THREW_EXCEPTION;
        }
        LOG_INFO("Detection Event '" << detectionEvent 
            << "' was removed from Reporter '" << reporter << "' successfully");

        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::ReporterDetectionEventRemoveAll(const char* reporter)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, reporter);

        try
        {
            DSL_REPORTER_PTR pReporterBintr = 
                std::dynamic_pointer_cast<ReporterBintr>(m_components[reporter]);

            pReporterBintr->RemoveAllChildren();
        }
        catch(...)
        {
            LOG_ERROR("Reporter '" << reporter 
                << "' threw an exception removing All Detection Events");
            return DSL_RESULT_REPORTER_THREW_EXCEPTION;
        }
        LOG_INFO("All Detection Events removed from Reporter '" << reporter << "' successfully");

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
        LOG_INFO("new OFV '" << name << "' created successfully");

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
        LOG_INFO("new OSD '" << name << "' created successfully");

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

    DslReturnType Services::OsdRedactionEnabledGet(const char* name, boolean* enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            *enabled = osdBintr->GetRedactionEnabled();
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception getting Redaction enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OsdRedactionEnabledSet(const char* name, boolean enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            if (!osdBintr->SetRedactionEnabled(enabled))
            {
                LOG_ERROR("OSD '" << name << "' failed to set Redaction enabled");
                return DSL_RESULT_OSD_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception setting Redaction enabled");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OsdRedactionClassAdd(const char* name, int classId, 
        double red, double green, double blue, double alpha)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            if (red > 1.0 or green > 1.0 or blue > 1.0 or alpha > 1.0)
            {
                LOG_ERROR("Invalid Redaction color param passed to OSD '" << name << "'");
                return DSL_RESULT_OSD_COLOR_PARAM_INVALID;
            }
            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            if (!osdBintr->AddRedactionClass(classId, red, green, blue, alpha))
            {
                LOG_ERROR("OSD '" << name << "' failed to add Redaction Class");
                return DSL_RESULT_OSD_REDACTION_CLASS_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception adding Redaction Class");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::OsdRedactionClassRemove(const char* name, int classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            if (!osdBintr->RemoveRedactionClass(classId))
            {
                LOG_ERROR("OSD '" << name << "' failed to remove Redaction Class");
                return DSL_RESULT_OSD_REDACTION_CLASS_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception removing Redaction Class");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OsdBatchMetaHandlerAdd(const char* name, uint pad, 
        dsl_batch_meta_handler_cb handler, void* user_data)
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

            if (!pOsdBintr->AddBatchMetaHandler(pad, handler, user_data))
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

    DslReturnType  Services::OsdKittiOutputEnabledSet(const char* name, boolean enabled,
        const char* file)    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);
            
            DSL_OSD_PTR pOsdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            if (!pOsdBintr->SetKittiOutputEnabled(enabled, file))
            {
                LOG_ERROR("Invalid Kitti file path " << file << "for Tracker '" << name << "'");
                return DSL_RESULT_OSD_SET_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Tracker '" << name << "' threw an exception setting Kitti output enabled");
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
        LOG_INFO("new BRANCH '" << name << "' created successfully");

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
        LOG_INFO("new PIPELINE '" << name << "' created successfully");

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
            m_returnValueToString[0xFFFFFFFF].c_str();
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
            m_stateValueToString[0xFFFFFFFF].c_str();
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
        m_stateValueToString[0xFFFFFFFF] = L"Invalid DSL_STATE Value";

        m_returnValueToString[DSL_RESULT_SUCCESS] = L"DSL_RESULT_SUCCESS";
        m_returnValueToString[DSL_RESULT_COMPONENT_RESULT] = L"DSL_RESULT_COMPONENT_RESULT";
        m_returnValueToString[DSL_RESULT_COMPONENT_NAME_NOT_UNIQUE] = L"DSL_RESULT_COMPONENT_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_COMPONENT_NAME_NOT_FOUND] = L"DSL_RESULT_COMPONENT_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_COMPONENT_NAME_BAD_FORMAT] = L"DSL_RESULT_COMPONENT_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_COMPONENT_IN_USE] = L"DSL_RESULT_COMPONENT_IN_USE";
        m_returnValueToString[DSL_RESULT_COMPONENT_NOT_USED_BY_PIPELINE] = L"DSL_RESULT_COMPONENT_NOT_USED_BY_PIPELINE";
        m_returnValueToString[DSL_RESULT_COMPONENT_NOT_USED_BY_BRANCH] = L"DSL_RESULT_COMPONENT_NOT_USED_BY_BRANCH";
        m_returnValueToString[DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE] = L"DSL_RESULT_COMPONENT_NOT_THE_CORRECT_TYPE";
        m_returnValueToString[DSL_RESULT_COMPONENT_SET_GPUID_FAILED] = L"DSL_RESULT_COMPONENT_SET_GPUID_FAILED";
        m_returnValueToString[DSL_RESULT_SOURCE_RESULT] = L"DSL_RESULT_SOURCE_RESULT";
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
        m_returnValueToString[DSL_RESULT_DEWARPER_RESULT] = L"DSL_RESULT_DEWARPER_RESULT";
        m_returnValueToString[DSL_RESULT_DEWARPER_NAME_NOT_UNIQUE] = L"DSL_RESULT_DEWARPER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_DEWARPER_NAME_NOT_FOUND] = L"DSL_RESULT_DEWARPER_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_DEWARPER_NAME_BAD_FORMAT] = L"DSL_RESULT_DEWARPER_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_DEWARPER_THREW_EXCEPTION] = L"DSL_RESULT_DEWARPER_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_DEWARPER_CONFIG_FILE_NOT_FOUND] = L"DSL_RESULT_DEWARPER_CONFIG_FILE_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_TRACKER_RESULT] = L"DSL_RESULT_TRACKER_RESULT";
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
        m_returnValueToString[DSL_RESULT_REPORTER_NAME_NOT_UNIQUE] = L"DSL_RESULT_REPORTER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_REPORTER_NAME_NOT_UNIQUE] = L"DSL_RESULT_REPORTER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_REPORTER_NAME_NOT_FOUND] = L"DSL_RESULT_REPORTER_NAME_NOT_FOUND";
        m_returnValueToString[DSL_RESULT_REPORTER_NAME_NOT_UNIQUE] = L"DSL_RESULT_REPORTER_NAME_NOT_UNIQUE";
        m_returnValueToString[DSL_RESULT_REPORTER_NAME_BAD_FORMAT] = L"DSL_RESULT_REPORTER_NAME_BAD_FORMAT";
        m_returnValueToString[DSL_RESULT_REPORTER_THREW_EXCEPTION] = L"DSL_RESULT_REPORTER_THREW_EXCEPTION";
        m_returnValueToString[DSL_RESULT_REPORTER_IS_IN_USE] = L"DSL_RESULT_REPORTER_IS_IN_USE";
        m_returnValueToString[DSL_RESULT_REPORTER_SET_FAILED] = L"DSL_RESULT_REPORTER_SET_FAILED";
        m_returnValueToString[DSL_RESULT_REPORTER_EVENT_ADD_FAILED] = L"DSL_RESULT_REPORTER_EVENT_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_REPORTER_EVENT_REMOVE_FAILED] = L"DSL_RESULT_REPORTER_EVENT_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_REPORTER_COMPONENT_IS_NOT_REPORTER] = L"DSL_RESULT_REPORTER_COMPONENT_IS_NOT_REPORTER";
        m_returnValueToString[DSL_RESULT_SINK_RESULT] = L"DSL_RESULT_SINK_RESULT";
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
        m_returnValueToString[DSL_RESULT_OSD_RESULT] = L"DSL_RESULT_OSD_RESULT";
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
        m_returnValueToString[DSL_RESULT_OSD_REDACTION_CLASS_ADD_FAILED] = L"DSL_RESULT_OSD_REDACTION_CLASS_ADD_FAILED";
        m_returnValueToString[DSL_RESULT_OSD_REDACTION_CLASS_REMOVE_FAILED] = L"DSL_RESULT_OSD_REDACTION_CALSS_REMOVE_FAILED";
        m_returnValueToString[DSL_RESULT_GIE_RESULT] = L"DSL_RESULT_GIE_RESULT";
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
        m_returnValueToString[DSL_RESULT_TEE_RESULT] = L"DSL_RESULT_TEE_RESULT";
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
        m_returnValueToString[DSL_RESULT_TILER_RESULT] = L"DSL_RESULT_TILER_RESULT";
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
        m_returnValueToString[0xFFFFFFFF] = L"Invalid DSL Result CODE";
    }

} // namespace 