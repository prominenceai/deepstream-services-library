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

#include <nvbufsurftransform.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/highgui/highgui.hpp"

#include "Dsl.h"
#include "DslServices.h"
#include "DslOdeType.h"
#include "DslOdeAction.h"

namespace DSL
{
    OdeAction::OdeAction(const char* name)
        : Base(name)
    {
    }

    OdeAction::~OdeAction()
    {
    }

    // ********************************************************************

    CallbackOdeAction::CallbackOdeAction(const char* name, 
        dsl_ode_occurrence_handler_cb clientHandler, void* clientData)
        : OdeAction(name)
        , m_clientHandler(clientHandler)
        , m_clientData(clientData)
    {
        LOG_FUNC();
    }

    CallbackOdeAction::~CallbackOdeAction()
    {
        LOG_FUNC();
    }
    
    void CallbackOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        DSL_ODE_TYPE_PTR pOdeType = std::dynamic_pointer_cast<OdeType>(pBaseType);

        m_clientHandler(pOdeType->s_eventCount, pOdeType->m_wName.c_str(),
            pFrameMeta, pObjectMeta, m_clientData);
    }

    // ********************************************************************

    CaptureOdeAction::CaptureOdeAction(const char* name, 
        uint captureType, const char* outdir)
        : OdeAction(name)
        , m_captureType(captureType)
        , m_outdir(outdir)
    {
        LOG_FUNC();
    }

    CaptureOdeAction::~CaptureOdeAction()
    {
        LOG_FUNC();
    }

    void CaptureOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // ensure that if we're capturing an Object, object data must be provided
        // i.e Object capture and Frame event action result in a NOP
        if ((m_captureType == DSL_CAPTURE_TYPE_OBJECT) and (!pObjectMeta))
        {
            return;
        }
        GstMapInfo inMapInfo = {0};

        if (!gst_buffer_map(pBuffer, &inMapInfo, GST_MAP_READ))
        {
            LOG_ERROR("ODE Capture Action '" << GetName() << "' failed to map gst buffer");
            gst_buffer_unmap(pBuffer, &inMapInfo);
            return;
        }
        NvBufSurface* surface = (NvBufSurface*)inMapInfo.data;

        DSL_ODE_TYPE_PTR pOdeType = std::dynamic_pointer_cast<OdeType>(pBaseType);
        
        std::string filespec = m_outdir + "/" + pOdeType->GetName() + "-" +
            std::to_string(pOdeType->s_eventCount) + ".jpg";

        NvBufSurfTransformRect src_rect = {0};
        NvBufSurfTransformRect dst_rect = {0};

        // capturing full frame or object only?
        if (m_captureType == DSL_CAPTURE_TYPE_FRAME)
        {
            src_rect.width = surface->surfaceList[0].width;
            src_rect.height = surface->surfaceList[0].height;
            dst_rect.width = surface->surfaceList[0].width;
            dst_rect.height = surface->surfaceList[0].height;
        }
        else
        {
            src_rect.top = pObjectMeta->rect_params.top;
            src_rect.left = pObjectMeta->rect_params.left;
            src_rect.width = pObjectMeta->rect_params.width; 
            src_rect.height = pObjectMeta->rect_params.height;
            dst_rect.width = pObjectMeta->rect_params.width; 
            dst_rect.height = pObjectMeta->rect_params.height;
        }
        NvBufSurfTransformParams bufSurfTransform;
        bufSurfTransform.src_rect = &src_rect;
        bufSurfTransform.dst_rect = &dst_rect;
        bufSurfTransform.transform_flag = NVBUFSURF_TRANSFORM_CROP_SRC |
            NVBUFSURF_TRANSFORM_CROP_DST;
        bufSurfTransform.transform_filter = NvBufSurfTransformInter_Default;

        NvBufSurface *dstSurface = NULL;

        NvBufSurfaceCreateParams bufSurfaceCreateParams;

        // An intermediate buffer for NV12/RGBA to BGR conversion
        bufSurfaceCreateParams.gpuId = surface->gpuId;
        bufSurfaceCreateParams.width = dst_rect.width;
        bufSurfaceCreateParams.height = dst_rect.height;
        bufSurfaceCreateParams.size = 0;
        bufSurfaceCreateParams.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
        bufSurfaceCreateParams.layout = NVBUF_LAYOUT_PITCH;
        bufSurfaceCreateParams.memType = NVBUF_MEM_DEFAULT;

        cudaError_t cudaError = cudaSetDevice(surface->gpuId);
        cudaStream_t cudaStream;
        cudaError = cudaStreamCreate(&cudaStream);

        int retval = NvBufSurfaceCreate(&dstSurface, surface->batchSize,
            &bufSurfaceCreateParams);	

        NvBufSurfTransformConfigParams bufSurfTransformConfigParams;
        NvBufSurfTransform_Error err;

        bufSurfTransformConfigParams.compute_mode = NvBufSurfTransformCompute_Default;
        bufSurfTransformConfigParams.gpu_id = surface->gpuId;
        bufSurfTransformConfigParams.cuda_stream = cudaStream;
        err = NvBufSurfTransformSetSessionParams (&bufSurfTransformConfigParams);

        NvBufSurfaceMemSet(dstSurface, 0, 0, 0);

        err = NvBufSurfTransform (surface, dstSurface, &bufSurfTransform);
        if (err != NvBufSurfTransformError_Success)
        {
            g_print ("NvBufSurfTransform failed with error %d while converting buffer\n", err);
        }

        NvBufSurfaceMap(dstSurface, 0, 0, NVBUF_MAP_READ);
        NvBufSurfaceSyncForCpu(dstSurface, 0, 0);

        cv::Mat bgr_frame = cv::Mat(cv::Size(bufSurfaceCreateParams.width,
            bufSurfaceCreateParams.height), CV_8UC3);

        cv::Mat in_mat = cv::Mat(bufSurfaceCreateParams.height, 
            bufSurfaceCreateParams.width, CV_8UC4, 
            dstSurface->surfaceList[0].mappedAddr.addr[0],
            dstSurface->surfaceList[0].pitch);

        cv::cvtColor (in_mat, bgr_frame, CV_RGBA2BGR);

        cv::imwrite(filespec.c_str(), bgr_frame);

        NvBufSurfaceUnMap(dstSurface, 0, 0);
        NvBufSurfaceDestroy(dstSurface);
        cudaStreamDestroy(cudaStream);
        gst_buffer_unmap(pBuffer, &inMapInfo);
    }

    // ********************************************************************

    DisplayOdeAction::DisplayOdeAction(const char* name)
        : OdeAction(name)
    {
        LOG_FUNC();
    }

    DisplayOdeAction::~DisplayOdeAction()
    {
        LOG_FUNC();
    }

    void DisplayOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        DSL_ODE_TYPE_PTR pOdeType = std::dynamic_pointer_cast<OdeType>(pBaseType);

    }

    // ********************************************************************

    LogOdeAction::LogOdeAction(const char* name)
        : OdeAction(name)
    {
        LOG_FUNC();
    }

    LogOdeAction::~LogOdeAction()
    {
        LOG_FUNC();
    }

    void LogOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        DSL_ODE_TYPE_PTR pOdeType = std::dynamic_pointer_cast<OdeType>(pBaseType);
        
        LOG_INFO("Event Name      : " << pOdeType->GetName());
        LOG_INFO("  Unique Id     : " << pOdeType->s_eventCount);
        LOG_INFO("  NTP Timestamp : " << pFrameMeta->ntp_timestamp);
        LOG_INFO("  Source Data   : ------------------------");
        LOG_INFO("    Id          : " << pFrameMeta->source_id);
        LOG_INFO("    Frame       : " << pFrameMeta->frame_num);
        LOG_INFO("    Width       : " << pFrameMeta->source_frame_width);
        LOG_INFO("    Heigh       : " << pFrameMeta->source_frame_height );
        LOG_INFO("  Object Data   : ------------------------");
        LOG_INFO("    Class Id    : " << pOdeType->m_classId );
        LOG_INFO("    Occurrences : " << pOdeType->m_occurrences );
        
        if (pObjectMeta)
        {
            LOG_INFO("    Tracking Id : " << pObjectMeta->object_id);
            LOG_INFO("    Label       : " << pObjectMeta->obj_label);
            LOG_INFO("    Confidence  : " << pObjectMeta->confidence);
            LOG_INFO("    Left        : " << pObjectMeta->rect_params.left);
            LOG_INFO("    Top         : " << pObjectMeta->rect_params.top);
            LOG_INFO("    Width       : " << pObjectMeta->rect_params.width);
            LOG_INFO("    Height      : " << pObjectMeta->rect_params.height);
        }
        LOG_INFO("  Min Criteria  : ------------------------");
        LOG_INFO("    Confidence  : " << pOdeType->m_minConfidence);
        LOG_INFO("    Frame Count : " << pOdeType->m_minFrameCountN
            << " out of " << pOdeType->m_minFrameCountD);
        LOG_INFO("    Width       : " << pOdeType->m_minWidth);
        LOG_INFO("    Height      : " << pOdeType->m_minHeight);
    }

    // ********************************************************************

    PauseOdeAction::PauseOdeAction(const char* name, const char* pipeline)
        : OdeAction(name)
        , m_pipeline(pipeline)
    {
        LOG_FUNC();
    }

    PauseOdeAction::~PauseOdeAction()
    {
        LOG_FUNC();
    }
    
    void PauseOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // Ignore the return value, errors will be logged 
        Services::GetServices()->PipelinePause(m_pipeline.c_str());
    }

    // ********************************************************************

    PrintOdeAction::PrintOdeAction(const char* name)
        : OdeAction(name)
    {
        LOG_FUNC();
    }

    PrintOdeAction::~PrintOdeAction()
    {
        LOG_FUNC();
    }

    void PrintOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        DSL_ODE_TYPE_PTR pOdeType = std::dynamic_pointer_cast<OdeType>(pBaseType);
        
        std::cout << "Event Name      : " << pOdeType->GetName() << "\n";
        std::cout << "  Unique Id     : " << pOdeType->s_eventCount << "\n";
        std::cout << "  NTP Timestamp : " << pFrameMeta->ntp_timestamp << "\n";
        std::cout << "  Source Data   : ------------------------" << "\n";
        std::cout << "    Id          : " << pFrameMeta->source_id << "\n";
        std::cout << "    Frame       : " << pFrameMeta->frame_num << "\n";
        std::cout << "    Width       : " << pFrameMeta->source_frame_width << "\n";
        std::cout << "    Heigh       : " << pFrameMeta->source_frame_height << "\n";
        std::cout << "  Object Data   : ------------------------" << "\n";
        std::cout << "    Class Id    : " << pOdeType->m_classId << "\n";
        std::cout << "    Occurrences : " << pOdeType->m_occurrences << "\n";

        if (pObjectMeta)
        {
            std::cout << "    Tracking Id : " << pObjectMeta->object_id << "\n";
            std::cout << "    Label       : " << pObjectMeta->obj_label << "\n";
            std::cout << "    Confidence  : " << pObjectMeta->confidence << "\n";
            std::cout << "    Left        : " << pObjectMeta->rect_params.left << "\n";
            std::cout << "    Top         : " << pObjectMeta->rect_params.top << "\n";
            std::cout << "    Width       : " << pObjectMeta->rect_params.width << "\n";
            std::cout << "    Height      : " << pObjectMeta->rect_params.height << "\n";
        }

        std::cout << "  Min Criteria  : ------------------------" << "\n";
        std::cout << "    Confidence  : " << pOdeType->m_minConfidence << "\n";
        std::cout << "    Frame Count : " << pOdeType->m_minFrameCountN
            << " out of " << pOdeType->m_minFrameCountD << "\n";
        std::cout << "    Width       : " << pOdeType->m_minWidth << "\n";
        std::cout << "    Height      : " << pOdeType->m_minHeight << "\n\n";
    }

    // ********************************************************************

    RedactOdeAction::RedactOdeAction(const char* name, double red, double green, double blue, double alpha)
        : OdeAction(name)
    {
        LOG_FUNC();
        m_backgroundColor.red = red;
        m_backgroundColor.green = green;
        m_backgroundColor.blue = blue;
        m_backgroundColor.alpha = alpha;
    }

    RedactOdeAction::~RedactOdeAction()
    {
        LOG_FUNC();

    }

    void RedactOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        // hide the OSD display text
        if (pObjectMeta->text_params.display_text)
        {
            pObjectMeta->text_params.set_bg_clr = 0;
            pObjectMeta->text_params.font_params.font_size = 0;
        }
        // shade in the background
        pObjectMeta->rect_params.border_width = 0;
        pObjectMeta->rect_params.has_bg_color = 1;
        pObjectMeta->rect_params.bg_color.red = m_backgroundColor.red;
        pObjectMeta->rect_params.bg_color.green = m_backgroundColor.green;
        pObjectMeta->rect_params.bg_color.blue = m_backgroundColor.blue;
        pObjectMeta->rect_params.bg_color.alpha = m_backgroundColor.alpha;
    }

    // ********************************************************************

    AddSinkOdeAction::AddSinkOdeAction(const char* name, 
        const char* pipeline, const char* sink)
        : OdeAction(name)
        , m_pipeline(pipeline)
        , m_sink(sink)
    {
        LOG_FUNC();
    }

    AddSinkOdeAction::~AddSinkOdeAction()
    {
        LOG_FUNC();
    }
    
    void AddSinkOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        Services::GetServices()->PipelineComponentAdd(m_pipeline.c_str(), m_sink.c_str());
    }

    // ********************************************************************

    RemoveSinkOdeAction::RemoveSinkOdeAction(const char* name, 
        const char* pipeline, const char* sink)
        : OdeAction(name)
        , m_pipeline(pipeline)
        , m_sink(sink)
    {
        LOG_FUNC();
    }

    RemoveSinkOdeAction::~RemoveSinkOdeAction()
    {
        LOG_FUNC();
    }
    
    void RemoveSinkOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        Services::GetServices()->PipelineComponentRemove(m_pipeline.c_str(), m_sink.c_str());
    }

    // ********************************************************************

    AddSourceOdeAction::AddSourceOdeAction(const char* name, 
        const char* pipeline, const char* source)
        : OdeAction(name)
        , m_pipeline(pipeline)
        , m_source(source)
    {
        LOG_FUNC();
    }

    AddSourceOdeAction::~AddSourceOdeAction()
    {
        LOG_FUNC();
    }
    
    void AddSourceOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        Services::GetServices()->PipelineComponentAdd(m_pipeline.c_str(), m_source.c_str());
    }

    // ********************************************************************

    RemoveSourceOdeAction::RemoveSourceOdeAction(const char* name, 
        const char* pipeline, const char* source)
        : OdeAction(name)
        , m_pipeline(pipeline)
        , m_source(source)
    {
        LOG_FUNC();
    }

    RemoveSourceOdeAction::~RemoveSourceOdeAction()
    {
        LOG_FUNC();
    }
    
    void RemoveSourceOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        Services::GetServices()->PipelineComponentRemove(m_pipeline.c_str(), m_source.c_str());
    }

    // ********************************************************************

    AddTypeOdeAction::AddTypeOdeAction(const char* name, 
        const char* odeType, const char* odeHandler)
        : OdeAction(name)
        , m_odeType(odeType)
        , m_odeHandler(odeHandler)
    {
        LOG_FUNC();
    }

    AddTypeOdeAction::~AddTypeOdeAction()
    {
        LOG_FUNC();
    }
    
    void AddTypeOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        Services::GetServices()->OdeHandlerTypeAdd(m_odeHandler.c_str(), m_odeType.c_str());
    }

    // ********************************************************************

    DisableTypeOdeAction::DisableTypeOdeAction(const char* name, const char* odeType)
        : OdeAction(name)
        , m_odeType(odeType)
    {
        LOG_FUNC();
    }

    DisableTypeOdeAction::~DisableTypeOdeAction()
    {
        LOG_FUNC();
    }
    
    void DisableTypeOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        Services::GetServices()->OdeTypeEnabledSet(m_odeType.c_str(), false);
    }

    // ********************************************************************

    EnableTypeOdeAction::EnableTypeOdeAction(const char* name, const char* odeType)
        : OdeAction(name)
        , m_odeType(odeType)
    {
        LOG_FUNC();
    }

    EnableTypeOdeAction::~EnableTypeOdeAction()
    {
        LOG_FUNC();
    }
    
    void EnableTypeOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        Services::GetServices()->OdeTypeEnabledSet(m_odeType.c_str(), true);
    }

    // ********************************************************************

    RemoveTypeOdeAction::RemoveTypeOdeAction(const char* name, 
        const char* odeType, const char* odeHandler)
        : OdeAction(name)
        , m_odeType(odeType)
        , m_odeHandler(odeHandler)
    {
        LOG_FUNC();
    }

    RemoveTypeOdeAction::~RemoveTypeOdeAction()
    {
        LOG_FUNC();
    }
    
    void RemoveTypeOdeAction::HandleOccurrence(DSL_BASE_PTR pBaseType, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        Services::GetServices()->OdeHandlerTypeRemove(m_odeHandler.c_str(), m_odeType.c_str());
    }

}    
    