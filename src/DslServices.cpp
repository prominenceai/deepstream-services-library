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
#include "DslServices.h"
#include "DslSourceBintr.h"
#include "DslGieBintr.h"
#include "DslTrackerBintr.h"
#include "DslTilerBintr.h"
#include "DslOsdBintr.h"
#include "DslSinkBintr.h"

// Single GST debug catagory initialization
GST_DEBUG_CATEGORY(GST_CAT_DSL);

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

DslReturnType dsl_source_sink_add(const wchar_t* source, const wchar_t* sink)
{
    std::wstring wstrSource(source);
    std::string cstrSource(wstrSource.begin(), wstrSource.end());
    std::wstring wstrSink(sink);
    std::string cstrSink(wstrSink.begin(), wstrSink.end());

    return DSL::Services::GetServices()->SourceSinkAdd(cstrSource.c_str(), cstrSink.c_str());
}

DslReturnType dsl_source_sink_remove(const wchar_t* source, const wchar_t* sink)
{
    std::wstring wstrSource(source);
    std::string cstrSource(wstrSource.begin(), wstrSource.end());
    std::wstring wstrSink(sink);
    std::string cstrSink(wstrSink.begin(), wstrSink.end());

    return DSL::Services::GetServices()->SourceSinkRemove(cstrSource.c_str(), cstrSink.c_str());
}

DslReturnType dsl_source_decode_dewarper_add(const wchar_t* source, const wchar_t* dewarper)
{
    std::wstring wstrSource(source);
    std::string cstrSource(wstrSource.begin(), wstrSource.end());
    std::wstring wstrDewarper(dewarper);
    std::string cstrDewarper(wstrDewarper.begin(), wstrDewarper.end());

    return DSL::Services::GetServices()->SourceDecodeDewarperAdd(cstrSource.c_str(), cstrDewarper.c_str());
}

DslReturnType dsl_source_decode_dewarper_remove(const wchar_t* source)
{
    std::wstring wstrSource(source);
    std::string cstrSource(wstrSource.begin(), wstrSource.end());

    return DSL::Services::GetServices()->SourceDecodeDewarperRemove(cstrSource.c_str());
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

DslReturnType dsl_gie_primary_batch_meta_handler_add(const wchar_t* name, uint pad, 
    dsl_batch_meta_handler_cb handler, void* user_data)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->PrimaryGieBatchMetaHandlerAdd(cstrName.c_str(), pad, handler, user_data);
}

DslReturnType dsl_gie_primary_batch_meta_handler_remove(const wchar_t* name, uint pad)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->PrimaryGieBatchMetaHandlerRemove(cstrName.c_str(), pad);
}

DslReturnType dsl_gie_secondary_new(const wchar_t* name, const wchar_t* infer_config_file,
    const wchar_t* model_engine_file, const wchar_t* infer_on_gie_name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrConfig(infer_config_file);
    std::string cstrConfig(wstrConfig.begin(), wstrConfig.end());
    std::wstring wstrEngine(model_engine_file);
    std::string cstrEngine(wstrEngine.begin(), wstrEngine.end());
    std::wstring wstrGie(infer_on_gie_name);
    std::string cstrGie(wstrGie.begin(), wstrGie.end());
    
    return DSL::Services::GetServices()->SecondaryGieNew(cstrName.c_str(), cstrConfig.c_str(),
        cstrEngine.c_str(), cstrGie.c_str());
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

DslReturnType dsl_tracker_batch_meta_handler_remove(const wchar_t* name, uint pad)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->TrackerBatchMetaHandlerRemove(cstrName.c_str(), pad);
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

//DslReturnType dsl_osd_clock_font_get(const wchar_t* name, const wchar_t** font, uint size);
//{
//    std::wstring wstrName(name);
//    std::string cstrName(wstrName.begin(), wstrName.end());
//
//    return DSL::Services::GetServices()->OsdClockFontsGet(cstrFont.c_str(), font, size);
//}

DslReturnType dsl_osd_clock_font_set(const wchar_t* name, const wchar_t* font, uint size)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    std::wstring wstrFont(font);
    std::string cstrFont(wstrFont.begin(), wstrFont.end());

    return DSL::Services::GetServices()->OsdClockFontSet(cstrFont.c_str(), cstrFont.c_str(), size);
}

DslReturnType dsl_osd_clock_color_get(const wchar_t* name, uint* red, uint* green, uint* blue)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockColorGet(cstrName.c_str(), red, green, blue);
}

DslReturnType dsl_osd_clock_color_set(const wchar_t* name, uint red, uint green, uint blue)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->OsdClockColorSet(cstrName.c_str(), red, green, blue);
}

DslReturnType dsl_osd_batch_meta_handler_add(const wchar_t* name, uint pad, 
    dsl_batch_meta_handler_cb handler, void* user_data)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->OsdBatchMetaHandlerAdd(cstrName.c_str(), pad, handler, user_data);
}

DslReturnType dsl_osd_batch_meta_handler_remove(const wchar_t* name, uint pad)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->OsdBatchMetaHandlerRemove(cstrName.c_str(), pad);
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

DslReturnType dsl_tiler_batch_meta_handler_remove(const wchar_t* name, uint pad)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());
    
    return DSL::Services::GetServices()->TilerBatchMetaHandlerRemove(cstrName.c_str(), pad);
}

DslReturnType dsl_sink_fake_new(const wchar_t* name)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkFakeNew(cstrName.c_str());
}

DslReturnType dsl_sink_overlay_new(const wchar_t* name,
    uint offsetX, uint offsetY, uint width, uint height)
{
    std::wstring wstrName(name);
    std::string cstrName(wstrName.begin(), wstrName.end());

    return DSL::Services::GetServices()->SinkOverlayNew(cstrName.c_str(), 
        offsetX, offsetY, width, height);
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

DslReturnType dsl_pipeline_new(const wchar_t* pipeline)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineNew(cstrPipeline.c_str());
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

DslReturnType dsl_pipeline_get_state(const wchar_t* pipeline)
{
    std::wstring wstrPipeline(pipeline);
    std::string cstrPipeline(wstrPipeline.begin(), wstrPipeline.end());

    return DSL::Services::GetServices()->PipelineGetState(cstrPipeline.c_str());
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

#define RETURN_IF_COMPONENT_IS_NOT_TRACKER(components, name) do \
{ \
    if (!components[name]->IsType(typeid(KtlTrackerBintr)) and  \
        !components[name]->IsType(typeid(IouTrackerBintr))) \
    { \
        LOG_ERROR("Component '" << name << "' is not a Tracker"); \
        return DSL_RESULT_TRACKER_COMPONENT_IS_NOT_TRACKER; \
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
            m_pInstatnce->_initMaps();
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
    
    DslReturnType Services::SourceSinkAdd(const char* source, const char* sink)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, source);
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, sink);
            RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, source);

            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[source]);
         
            DSL_SINK_PTR pSinkBintr = 
                std::dynamic_pointer_cast<SinkBintr>(m_components[sink]);
         
            if (!pSourceBintr->AddSinkBintr(pSinkBintr))
            {
                LOG_ERROR("Failed to add Sink '" << sink << "' to Source '" << source << "'");
                return DSL_RESULT_SOURCE_SINK_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Source '" << source << "' threw exception adding Sink");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SourceSinkRemove(const char* source, const char* sink)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, source);
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, sink);
            RETURN_IF_COMPONENT_IS_NOT_SOURCE(m_components, source);

            DSL_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<SourceBintr>(m_components[source]);
         
            DSL_SINK_PTR pSinkBintr = 
                std::dynamic_pointer_cast<SinkBintr>(m_components[sink]);
         
            if (!pSourceBintr->RemoveSinkBintr(pSinkBintr))
            {
                LOG_ERROR("Failed to remove Sink '" << sink << "' from Source '" << source << "'");
                return DSL_RESULT_SOURCE_SINK_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Source '" << source << "' threw exception removing Sink");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::SourceDecodeDewarperAdd(const char* source, const char* dewarper)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, source);
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, dewarper);
            RETURN_IF_COMPONENT_IS_NOT_DECODE_SOURCE(m_components, source);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, dewarper, DewarperBintr);

            DSL_DECODE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<DecodeSourceBintr>(m_components[source]);
         
            DSL_DEWARPER_PTR pDewarperBintr = 
                std::dynamic_pointer_cast<DewarperBintr>(m_components[dewarper]);
         
            if (!pSourceBintr->AddDewarperBintr(pDewarperBintr))
            {
                LOG_ERROR("Failed to add Dewarper '" << dewarper << "' to Decode Source '" << source << "'");
                return DSL_RESULT_SOURCE_DEWARPER_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Source '" << source << "' threw exception adding Dewarper");
            return DSL_RESULT_SOURCE_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SourceDecodeDewarperRemove(const char* source)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, source);
            RETURN_IF_COMPONENT_IS_NOT_DECODE_SOURCE(m_components, source);

            DSL_DECODE_SOURCE_PTR pSourceBintr = 
                std::dynamic_pointer_cast<DecodeSourceBintr>(m_components[source]);
         
            if (!pSourceBintr->RemoveDewarperBintr())
            {
                LOG_ERROR("Failed to remove Dewarper from Decode Source '" << source << "'");
                return DSL_RESULT_SOURCE_DEWARPER_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("Source '" << source << "' threw exception removing Dewarper");
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
            if (!pSourceBintr->Pause())
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

            if (!pSourceBintr->Play())
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
                LOG_ERROR("Primary GIE '" << name << "' already has a Batch Meta Handler");
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

    DslReturnType Services::PrimaryGieBatchMetaHandlerRemove(const char* name, uint pad)
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

            if (!pPrimaryGieBintr->RemoveBatchMetaHandler(pad))
            {
                LOG_ERROR("Primary GIE '" << name << "' has no Batch Meta Handler");
                return DSL_RESULT_GIE_HANDLER_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception removing Batch Meta Handle");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }
    
    DslReturnType Services::SecondaryGieNew(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, const char* inferOnGieName)
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
                inferConfigFile, modelEngineFile, inferOnGieName);
        }
        catch(...)
        {
            LOG_ERROR("New Primary GIE '" << name << "' threw exception on create");
            return DSL_RESULT_GIE_THREW_EXCEPTION;
        }
        LOG_INFO("new Secondary GIE '" << name << "' created successfully");

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

    DslReturnType Services::TrackerBatchMetaHandlerAdd(const char* name, uint pad, dsl_batch_meta_handler_cb handler, void* user_data)
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
                LOG_ERROR("Tracker '" << name << "' already has a Batch Meta Handler");
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

    DslReturnType Services::TrackerBatchMetaHandlerRemove(const char* name, uint pad)
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

            if (!pTrackerBintr->RemoveBatchMetaHandler(pad))
            {
                LOG_ERROR("Tracker '" << name << "' has no Batch Meta Handler");
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
            LOG_ERROR("Tiler New'" << name << "' threw exception on create");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
        LOG_INFO("new Tiler '" << name << "' created successfully");

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

    DslReturnType Services::TilerBatchMetaHandlerAdd(const char* name, uint pad, dsl_batch_meta_handler_cb handler, void* user_data)
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
                LOG_ERROR("Tiler '" << name << "' already has a Batch Meta Handler");
                return DSL_RESULT_TILER_HANDLER_ADD_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception adding Batch Meta Handler");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::TilerBatchMetaHandlerRemove(const char* name, uint pad)
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

            if (!pTilerBintr->RemoveBatchMetaHandler(pad))
            {
                LOG_ERROR("Tiler '" << name << "' has no Batch Meta Handler");
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
            return DSL_RESULT_SINK_THREW_EXCEPTION;
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

    DslReturnType Services::OsdClockColorGet(const char* name, uint* red, uint* green, uint* blue)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, OsdBintr);

            DSL_OSD_PTR osdBintr = 
                std::dynamic_pointer_cast<OsdBintr>(m_components[name]);

            osdBintr->GetClockColor(red, green, blue);
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception getting clock font");
            return DSL_RESULT_OSD_THREW_EXCEPTION;
        }
        return DSL_RESULT_SUCCESS;
    }

    DslReturnType Services::OsdClockColorSet(const char* name, uint red, uint green, uint blue)
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
            if (!osdBintr->SetClockColor(red, green, blue))
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
    
    DslReturnType Services::OsdBatchMetaHandlerAdd(const char* name, uint pad, dsl_batch_meta_handler_cb handler, void* user_data)
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

    DslReturnType Services::OsdBatchMetaHandlerRemove(const char* name, uint pad)
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

            if (!pOsdBintr->RemoveBatchMetaHandler(pad))
            {
                LOG_ERROR("OSD '" << name << "' has no Batch Meta Handler");
                return DSL_RESULT_OSD_HANDLER_REMOVE_FAILED;
            }
        }
        catch(...)
        {
            LOG_ERROR("OSD '" << name << "' threw an exception removing Batch Meta Handle");
            return DSL_RESULT_TILER_THREW_EXCEPTION;
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
    
    DslReturnType Services::SinkOverlayNew(const char* name, 
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
            m_components[name] = DSL_OVERLAY_SINK_NEW(name, offsetX, offsetY, width, height);
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
        if (container > DSL_CONTAINER_MK4)
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
        
        for (auto const& imap: m_components)
        {
            if (imap.second->IsInUse())
            {
                LOG_ERROR("Component '" << imap.second->GetName() << "' is currently in use");
                return DSL_RESULT_COMPONENT_IN_USE;
            }
        }
        LOG_DEBUG("All components are un-owned and will be deleted");

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
                return DSL_RESULT_PIPELINE_SOURCE_MAX_IN_USE_REACED;
            }

            if (IsSinkComponent(component) and (GetNumSinksInUse() == m_sinkNumInUseMax))
            {
                LOG_ERROR("Adding Sink '" << component << "' to Pipeline '" << pipeline << 
                    "' would exceed the maximum num-in-use limit");
                return DSL_RESULT_PIPELINE_SINK_MAX_IN_USE_REACED;
            }

            m_components[component]->AddToParent(m_pipelines[pipeline]);
            LOG_INFO("Component '" << component 
                << "' was added to Pipeline '" << pipeline << "' successfully");
        }
        catch(...)
        {
            LOG_ERROR("Pipeline '" << pipeline
                << "' threw exception adding component '" << component << "'");
            return DSL_RESULT_PIPELINE_COMPONENT_ADD_FAILED;
        }
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
            if (!m_pipelines[pipeline]->GetStreamMuxBatchProperties(batchSize, batchTimeout))
            {
                LOG_ERROR("Pipeline '" << pipeline
                    << "' failed to get the Stream Muxer Batch Properties");
                return DSL_RESULT_PIPELINE_STREAMMUX_GET_FAILED;
            }
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
    
    DslReturnType Services::PipelineGetState(const char* pipeline)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);
        RETURN_IF_PIPELINE_NAME_NOT_FOUND(m_pipelines, pipeline);

        return DSL_RESULT_API_NOT_IMPLEMENTED;
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

    void Services::_initMaps()
    {
        LOG_FUNC();
        
        m_mapParserTypes[DSL_SOURCE_CODEC_PARSER_H264] = "h264parse";
        m_mapParserTypes[DSL_SOURCE_CODEC_PARSER_H265] = "h265parse";
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

} // namespace 