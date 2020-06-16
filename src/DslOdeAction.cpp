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
#include "DslOdeTrigger.h"
#include "DslOdeAction.h"

#define MAX_DISPLAY_LEN 64

namespace DSL
{
    OdeAction::OdeAction(const char* name)
        : Base(name)
        , m_enabled(true)
    {
    }

    OdeAction::~OdeAction()
    {
    }

    bool OdeAction::GetEnabled()
    {
        LOG_FUNC();
        
        return m_enabled;
    }
    
    void OdeAction::SetEnabled(bool enabled)
    {
        LOG_FUNC();
        
        m_enabled = enabled;
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
    
    void CallbackOdeAction::HandleOccurrence(DSL_BASE_PTR pTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            DSL_ODE_TRIGGER_PTR pTrigger = std::dynamic_pointer_cast<OdeTrigger>(pTrigger);

            m_clientHandler(pTrigger->s_eventCount, pTrigger->m_wName.c_str(),
                pFrameMeta, pObjectMeta, m_clientData);
        }
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

    void CaptureOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (!m_enabled)
        {
            return;
        }
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

        DSL_ODE_TRIGGER_PTR pTrigger = std::dynamic_pointer_cast<OdeTrigger>(pOdeTrigger);
        
        std::string filespec = m_outdir + "/" + pTrigger->GetName() + "-" +
            std::to_string(pTrigger->s_eventCount) + ".jpg";

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

    DisplayOdeAction::DisplayOdeAction(const char* name, 
        uint offsetX, uint offsetY, bool offsetYWithClassId)
        : OdeAction(name)
        , m_offsetX(offsetX)
        , m_offsetY(offsetX)
        , m_offsetYWithClassId(offsetYWithClassId)
    {
        LOG_FUNC();
    }

    DisplayOdeAction::~DisplayOdeAction()
    {
        LOG_FUNC();
    }

    void DisplayOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            NvDsBatchMeta* batchMeta = gst_buffer_get_nvds_batch_meta(pBuffer);
            
            DSL_ODE_TRIGGER_PTR pTrigger = std::dynamic_pointer_cast<OdeTrigger>(pOdeTrigger);
            
            NvDsDisplayMeta* pDisplayMeta = nvds_acquire_display_meta_from_pool(batchMeta);
            pDisplayMeta->num_labels = 1;

            NvOSD_TextParams *pTextParams = &pDisplayMeta->text_params[0];
            pTextParams->display_text = (gchar*) g_malloc0(MAX_DISPLAY_LEN);
            
            std::string test = pTrigger->GetName() + " = " + std::to_string(pTrigger->m_occurrences);
            test.copy(pTextParams->display_text, MAX_DISPLAY_LEN, 0);

            // Setup X and Y display offsets
            pTextParams->x_offset = m_offsetX;
            pTextParams->y_offset = m_offsetY;
            
            // Typically set if action is shared by multiple ODE Triggers/ClassId's 
            if (m_offsetYWithClassId)
            {
                pTextParams->y_offset += pTrigger->m_classId * 30 + 2;
            }

            // Font, font-size, font-color
            pTextParams->font_params.font_name = (gchar *) "Serif";
            pTextParams->font_params.font_size = 10;
            pTextParams->font_params.font_color.red = 1.0;
            pTextParams->font_params.font_color.green = 1.0;
            pTextParams->font_params.font_color.blue = 1.0;
            pTextParams->font_params.font_color.alpha = 1.0;

            // Text background color
            pTextParams->set_bg_clr = 1;
            pTextParams->text_bg_clr.red = 0.0;
            pTextParams->text_bg_clr.green = 0.0;
            pTextParams->text_bg_clr.blue = 0.0;
            pTextParams->text_bg_clr.alpha = 1.0;
            
            nvds_add_display_meta_to_frame(pFrameMeta, pDisplayMeta);
        }
    }

    // ********************************************************************

    FillOdeAction::FillOdeAction(const char* name, double red, double green, double blue, double alpha)
        : OdeAction(name)
    {
        LOG_FUNC();
        m_backgroundColor.red = red;
        m_backgroundColor.green = green;
        m_backgroundColor.blue = blue;
        m_backgroundColor.alpha = alpha;
        LOG_INFO("Setting Area '" << name << "to: red = " << red << " green = " 
            << green << " blue = " << blue << " alpha = " << alpha);        
    }

    FillOdeAction::~FillOdeAction()
    {
        LOG_FUNC();

    }

    void FillOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled and pObjectMeta)
        {
            pObjectMeta->rect_params.has_bg_color = 1;
            pObjectMeta->rect_params.bg_color = m_backgroundColor;
        }
    }

    // ********************************************************************

    HideOdeAction::HideOdeAction(const char* name, bool text, bool border)
        : OdeAction(name)
        , m_hideText(text)
        , m_hideBorder(border)
    {
        LOG_FUNC();
    }

    HideOdeAction::~HideOdeAction()
    {
        LOG_FUNC();
    }

    void HideOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled and pObjectMeta)
        {
            if (m_hideText and (pObjectMeta->text_params.display_text))
            {
                pObjectMeta->text_params.set_bg_clr = 0;
                pObjectMeta->text_params.font_params.font_size = 0;
            }
            if (m_hideBorder)
            {
                pObjectMeta->rect_params.border_width = 0;
            }
        }
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

    void LogOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            DSL_ODE_TRIGGER_PTR pTrigger = std::dynamic_pointer_cast<OdeTrigger>(pOdeTrigger);
            
            LOG_INFO("Trigger Name    : " << pTrigger->GetName());
            LOG_INFO("  Unique ODE Id : " << pTrigger->s_eventCount);
            LOG_INFO("  NTP Timestamp : " << pFrameMeta->ntp_timestamp);
            LOG_INFO("  Source Data   : ------------------------");
            LOG_INFO("    Id          : " << pFrameMeta->source_id);
            LOG_INFO("    Frame       : " << pFrameMeta->frame_num);
            LOG_INFO("    Width       : " << pFrameMeta->source_frame_width);
            LOG_INFO("    Heigh       : " << pFrameMeta->source_frame_height );
            LOG_INFO("  Object Data   : ------------------------");
            LOG_INFO("    Class Id    : " << pTrigger->m_classId );
            LOG_INFO("    Occurrences : " << pTrigger->m_occurrences );
            
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
            LOG_INFO("    Confidence  : " << pTrigger->m_minConfidence);
            LOG_INFO("    Frame Count : " << pTrigger->m_minFrameCountN
                << " out of " << pTrigger->m_minFrameCountD);
            LOG_INFO("    Width       : " << pTrigger->m_minWidth);
            LOG_INFO("    Height      : " << pTrigger->m_minHeight);
        }
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
    
    void PauseOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->PipelinePause(m_pipeline.c_str());
        }
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

    void PrintOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            DSL_ODE_TRIGGER_PTR pTrigger = std::dynamic_pointer_cast<OdeTrigger>(pOdeTrigger);
            
            std::cout << "Trigger Name    : " << pTrigger->GetName() << "\n";
            std::cout << "  Unique ODE Id : " << pTrigger->s_eventCount << "\n";
            std::cout << "  NTP Timestamp : " << pFrameMeta->ntp_timestamp << "\n";
            std::cout << "  Source Data   : ------------------------" << "\n";
            std::cout << "    Id          : " << pFrameMeta->source_id << "\n";
            std::cout << "    Frame       : " << pFrameMeta->frame_num << "\n";
            std::cout << "    Width       : " << pFrameMeta->source_frame_width << "\n";
            std::cout << "    Heigh       : " << pFrameMeta->source_frame_height << "\n";
            std::cout << "  Object Data   : ------------------------" << "\n";
            std::cout << "    Class Id    : " << pTrigger->m_classId << "\n";
            std::cout << "    Occurrences : " << pTrigger->m_occurrences << "\n";

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
            std::cout << "    Confidence  : " << pTrigger->m_minConfidence << "\n";
            std::cout << "    Frame Count : " << pTrigger->m_minFrameCountN
                << " out of " << pTrigger->m_minFrameCountD << "\n";
            std::cout << "    Width       : " << pTrigger->m_minWidth << "\n";
            std::cout << "    Height      : " << pTrigger->m_minHeight << "\n\n";
        }
    }

    // ********************************************************************

    RedactOdeAction::RedactOdeAction(const char* name)
        : OdeAction(name)
    {
        LOG_FUNC();
    }

    RedactOdeAction::~RedactOdeAction()
    {
        LOG_FUNC();

    }

    void RedactOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer,
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled and pObjectMeta)
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
            pObjectMeta->rect_params.bg_color.red = 0.0;
            pObjectMeta->rect_params.bg_color.green = 0.0;
            pObjectMeta->rect_params.bg_color.blue = 0.0;
            pObjectMeta->rect_params.bg_color.alpha = 1.0;
        }
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
    
    void AddSinkOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            Services::GetServices()->PipelineComponentAdd(m_pipeline.c_str(), m_sink.c_str());
        }
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
    
    void RemoveSinkOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->PipelineComponentRemove(m_pipeline.c_str(), m_sink.c_str());
        }
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
    
    void AddSourceOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->PipelineComponentAdd(m_pipeline.c_str(), m_source.c_str());
        }
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
    
    void RemoveSourceOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->PipelineComponentRemove(m_pipeline.c_str(), m_source.c_str());
        }
    }

    // ********************************************************************

    AddTriggerOdeAction::AddTriggerOdeAction(const char* name, 
        const char* handler, const char* trigger)
        : OdeAction(name)
        , m_handler(handler)
        , m_trigger(trigger)
    {
        LOG_FUNC();
    }

    AddTriggerOdeAction::~AddTriggerOdeAction()
    {
        LOG_FUNC();
    }
    
    void AddTriggerOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeHandlerTriggerAdd(m_handler.c_str(), m_trigger.c_str());
        }
    }

    // ********************************************************************

    DisableTriggerOdeAction::DisableTriggerOdeAction(const char* name, const char* trigger)
        : OdeAction(name)
        , m_trigger(trigger)
    {
        LOG_FUNC();
    }

    DisableTriggerOdeAction::~DisableTriggerOdeAction()
    {
        LOG_FUNC();
    }
    
    void DisableTriggerOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeTriggerEnabledSet(m_trigger.c_str(), false);
        }
    }

    // ********************************************************************

    EnableTriggerOdeAction::EnableTriggerOdeAction(const char* name, const char* trigger)
        : OdeAction(name)
        , m_trigger(trigger)
    {
        LOG_FUNC();
    }

    EnableTriggerOdeAction::~EnableTriggerOdeAction()
    {
        LOG_FUNC();
    }
    
    void EnableTriggerOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeTriggerEnabledSet(m_trigger.c_str(), true);
        }
    }

    // ********************************************************************

    RemoveTriggerOdeAction::RemoveTriggerOdeAction(const char* name, 
        const char* handler, const char* trigger)
        : OdeAction(name)
        , m_handler(handler)
        , m_trigger(trigger)
    {
        LOG_FUNC();
    }

    RemoveTriggerOdeAction::~RemoveTriggerOdeAction()
    {
        LOG_FUNC();
    }
    
    void RemoveTriggerOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeHandlerTriggerRemove(m_handler.c_str(), m_trigger.c_str());
        }
    }

    // ********************************************************************

    AddActionOdeAction::AddActionOdeAction(const char* name, 
        const char* trigger, const char* action)
        : OdeAction(name)
        , m_trigger(trigger)
        , m_action(action)
    {
        LOG_FUNC();
    }

    AddActionOdeAction::~AddActionOdeAction()
    {
        LOG_FUNC();
    }
    
    void AddActionOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeTriggerActionAdd(m_trigger.c_str(), m_action.c_str());
        }
    }

    // ********************************************************************

    DisableActionOdeAction::DisableActionOdeAction(const char* name, const char* action)
        : OdeAction(name)
        , m_action(action)
    {
        LOG_FUNC();
    }

    DisableActionOdeAction::~DisableActionOdeAction()
    {
        LOG_FUNC();
    }
    
    void DisableActionOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeActionEnabledSet(m_action.c_str(), false);
        }
    }

    // ********************************************************************

    EnableActionOdeAction::EnableActionOdeAction(const char* name, const char* action)
        : OdeAction(name)
        , m_action(action)
    {
        LOG_FUNC();
    }

    EnableActionOdeAction::~EnableActionOdeAction()
    {
        LOG_FUNC();
    }
    
    void EnableActionOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeActionEnabledSet(m_action.c_str(), true);
        }
    }

    // ********************************************************************

    RemoveActionOdeAction::RemoveActionOdeAction(const char* name, 
        const char* trigger, const char* action)
        : OdeAction(name)
        , m_trigger(trigger)
        , m_action(action)
    {
        LOG_FUNC();
    }

    RemoveActionOdeAction::~RemoveActionOdeAction()
    {
        LOG_FUNC();
    }
    
    void RemoveActionOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeTriggerActionRemove(m_trigger.c_str(), m_action.c_str());
        }
    }

    // ********************************************************************

    AddAreaOdeAction::AddAreaOdeAction(const char* name, 
        const char* trigger, const char* area)
        : OdeAction(name)
        , m_trigger(trigger)
        , m_area(area)
    {
        LOG_FUNC();
    }

    AddAreaOdeAction::~AddAreaOdeAction()
    {
        LOG_FUNC();
    }
    
    void AddAreaOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeTriggerAreaAdd(m_trigger.c_str(), m_area.c_str());
        }
    }
    
    // ********************************************************************

    RemoveAreaOdeAction::RemoveAreaOdeAction(const char* name, 
        const char* trigger, const char* area)
        : OdeAction(name)
        , m_trigger(trigger)
        , m_area(area)
    {
        LOG_FUNC();
    }

    RemoveAreaOdeAction::~RemoveAreaOdeAction()
    {
        LOG_FUNC();
    }
    
    void RemoveAreaOdeAction::HandleOccurrence(DSL_BASE_PTR pOdeTrigger, GstBuffer* pBuffer, 
        NvDsFrameMeta* pFrameMeta, NvDsObjectMeta* pObjectMeta)
    {
        if (m_enabled)
        {
            // Ignore the return value, errors will be logged 
            Services::GetServices()->OdeTriggerAreaRemove(m_trigger.c_str(), m_area.c_str());
        }
    }
    
}    
    