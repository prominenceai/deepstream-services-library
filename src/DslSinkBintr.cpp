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
#include "DslSinkBintr.h"
#include "DslBranchBintr.h"

namespace DSL
{

    SinkBintr::SinkBintr(const char* name)
        : Bintr(name)
        , m_isWindowCapable(false)
    {
        LOG_FUNC();

        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "sink-bin-queue");
        AddChild(m_pQueue);
        m_pQueue->AddGhostPadToParent("sink");
    }

    SinkBintr::~SinkBintr()
    {
        LOG_FUNC();
    }

    bool SinkBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' Sink to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddSinkBintr(shared_from_this());
    }


    bool SinkBintr::IsParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        LOG_WARN("**************************************");
        
        // check if 'this' Sink is child of Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            IsSinkBintrChild(std::dynamic_pointer_cast<SinkBintr>(shared_from_this()));
    }

    bool SinkBintr::RemoveFromParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        if (!IsParent(pParentBintr))
        {
            LOG_ERROR("Sink '" << GetName() << "' is not a child of Pipeline '" << pParentBintr->GetName() << "'");
            return false;
        }
        // remove 'this' Sink from the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            RemoveSinkBintr(std::dynamic_pointer_cast<SinkBintr>(shared_from_this()));
    }

    bool SinkBintr::IsWindowCapable()
    {
        LOG_FUNC();
        
        return m_isWindowCapable;
    }
    
    bool SinkBintr::LinkToSource(DSL_NODETR_PTR pTee)
    {
        LOG_FUNC();

        std::string srcPadName = "src_" + std::to_string(m_uniqueId);

        LOG_INFO("Linking Sink '" << GetName() << "' to Pad '" << srcPadName
            << "' for Tee '" << pTee->GetName() << "'");
        
        m_pGstStaticSinkPad = gst_element_get_static_pad(GetGstElement(), "sink");
        if (!m_pGstStaticSinkPad)
        {
            LOG_ERROR("Failed to get Static Sink Pad for SinkBintr '" << GetName() << "'");
            return false;
        }

        GstPad* pRequestedSourcePad(NULL);

        // NOTE: important to use the correct request pad name based on the element type
        // Cast the base DSL_BASE_PTR to DSL_ELEMENTR_PTR so we can query the factory type 
        DSL_ELEMENT_PTR pTeeElementr = 
            std::dynamic_pointer_cast<Elementr>(pTee);

        if (pTeeElementr->IsFactoryName("nvstreamdemux"))
        {
            pRequestedSourcePad = gst_element_get_request_pad(pTee->GetGstElement(), srcPadName.c_str());
        }
        else // standard "Tee"
        {
            pRequestedSourcePad = gst_element_get_request_pad(pTee->GetGstElement(), "src_%u");
        }
            
        if (!pRequestedSourcePad)
        {
            LOG_ERROR("Failed to get Tee source Pad for SinkBintr '" << GetName() <<"'");
            return false;
        }

        m_pGstRequestedSourcePads[srcPadName] = pRequestedSourcePad;

        return Bintr::LinkToSource(pTee);
        
    }
    
    bool SinkBintr::UnlinkFromSource()
    {
        LOG_FUNC();
        
        // If we're not currently linked to the Tee
        if (!IsLinkedToSource())
        {
            LOG_ERROR("SinkBintr '" << GetName() << "' is not in a Linked state");
            return false;
        }

        std::string srcPadName = "src_" + std::to_string(m_uniqueId);

        LOG_INFO("Unlinking and releasing requested Source Pad for Sink Tee " << GetName());
        
        gst_pad_send_event(m_pGstStaticSinkPad, gst_event_new_eos());
        if (!gst_pad_unlink(m_pGstRequestedSourcePads[srcPadName], m_pGstStaticSinkPad))
        {
            LOG_ERROR("SinkBintr '" << GetName() << "' failed to unlink from MultiSinks Tee");
            return false;
        }
        gst_element_release_request_pad(GetSource()->GetGstElement(), m_pGstRequestedSourcePads[srcPadName]);
        gst_object_unref(m_pGstRequestedSourcePads[srcPadName]);
                
        m_pGstRequestedSourcePads.erase(srcPadName);
        
        return Nodetr::UnlinkFromSource();
    }

    //-------------------------------------------------------------------------

    FakeSinkBintr::FakeSinkBintr(const char* name)
        : SinkBintr(name)
        , m_sync(TRUE)
        , m_async(FALSE)
        , m_qos(TRUE)
    {
        LOG_FUNC();
        
        m_isWindowCapable = false;

        m_pFakeSink = DSL_ELEMENT_NEW(NVDS_ELEM_SINK_FAKESINK, "sink-bin-fake");
        m_pFakeSink->SetAttribute("sync", m_sync);
        m_pFakeSink->SetAttribute("max-lateness", -1);
        m_pFakeSink->SetAttribute("async", m_async);
        m_pFakeSink->SetAttribute("qos", m_qos);
        
        AddChild(m_pFakeSink);
    }
    
    FakeSinkBintr::~FakeSinkBintr()
    {
        LOG_FUNC();
    
        if (IsLinked())
        {    
            UnlinkAll();
        }
    }

    bool FakeSinkBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("FakeSinkBintr '" << m_name << "' is already linked");
            return false;
        }
        // Single element, nothing to link
        m_isLinked = true;
        return true;
    }
    
    void FakeSinkBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("FakeSinkBintr '" << m_name << "' is not linked");
            return;
        }
        m_isLinked = false;
    }
    
    //-------------------------------------------------------------------------

    OverlaySinkBintr::OverlaySinkBintr(const char* name, uint overlayId, uint displayId, 
        uint depth, uint offsetX, uint offsetY, uint width, uint height)
        : SinkBintr(name)
        , m_sync(TRUE)
        , m_async(FALSE)
        , m_qos(TRUE)
        , m_overlayId(overlayId)
        , m_displayId(displayId)
        , m_depth(depth)
        , m_offsetX(offsetX)
        , m_offsetY(offsetY)
        , m_width(width)
        , m_height(height)
    {
        LOG_FUNC();
        
        m_isWindowCapable = false;

        m_pOverlay = DSL_ELEMENT_NEW(NVDS_ELEM_SINK_OVERLAY, "sink-bin-overlay");
        m_pOverlay->SetAttribute("overlay", m_overlayId);
        m_pOverlay->SetAttribute("display-id", m_displayId);
        m_pOverlay->SetAttribute("overlay-x", m_offsetX);
        m_pOverlay->SetAttribute("overlay-y", m_offsetY);
        m_pOverlay->SetAttribute("overlay-w", m_width);
        m_pOverlay->SetAttribute("overlay-h", m_height);
        m_pOverlay->SetAttribute("sync", m_sync);
        m_pOverlay->SetAttribute("max-lateness", -1);
        m_pOverlay->SetAttribute("async", m_async);
        m_pOverlay->SetAttribute("qos", m_qos);
        
        AddChild(m_pOverlay);
    }
    
    OverlaySinkBintr::~OverlaySinkBintr()
    {
        LOG_FUNC();
    
        if (IsLinked())
        {    
            UnlinkAll();
        }
    }

    bool OverlaySinkBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("OverlaySinkBintr '" << m_name << "' is already linked");
            return false;
        }
        if (!m_pQueue->LinkToSink(m_pOverlay))
        {
            return false;
        }
        m_isLinked = true;
        return true;
    }
    
    void OverlaySinkBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("OverlaySinkBintr '" << m_name << "' is not linked");
            return;
        }
        m_pQueue->UnlinkFromSink();
        m_isLinked = false;
    }

    int OverlaySinkBintr::GetDisplayId()
    {
        LOG_FUNC();
        
        return m_displayId;
    }
    
    bool OverlaySinkBintr::SetDisplayId(int id)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Dimensions for OverlaySinkBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_displayId = id;
        m_pOverlay->SetAttribute("display-id", m_displayId);
        
        return true;
    }
    
    void  OverlaySinkBintr::GetOffsets(uint* offsetX, uint* offsetY)
    {
        LOG_FUNC();
        
        *offsetX = m_offsetX;
        *offsetY = m_offsetY;
    }
    
    bool OverlaySinkBintr::SetOffsets(uint offsetX, uint offsetY)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Dimensions for OverlaySinkBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_offsetX = offsetX;
        m_offsetY = offsetY;

        m_pOverlay->SetAttribute("overlay-x", m_offsetX);
        m_pOverlay->SetAttribute("overlay-y", m_offsetY);
        
        return true;
    }

    void OverlaySinkBintr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_width;
        *height = m_height;
    }

    bool OverlaySinkBintr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Dimensions for OverlaySinkBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_width = width;
        m_height = height;

        m_pOverlay->SetAttribute("overlay-w", m_width);
        m_pOverlay->SetAttribute("overlay-h", m_height);
        
        return true;
    }
    
    //-------------------------------------------------------------------------

    WindowSinkBintr::WindowSinkBintr(const char* name, guint offsetX, guint offsetY, 
        guint width, guint height)
        : SinkBintr(name)
        , m_sync(TRUE)
        , m_async(FALSE)
        , m_qos(TRUE)
        , m_offsetX(offsetX)
        , m_offsetY(offsetY)
        , m_width(width)
        , m_height(height)
    {
        LOG_FUNC();
        
        m_isWindowCapable = true;

        m_pTransform = DSL_ELEMENT_NEW(NVDS_ELEM_EGLTRANSFORM, "sink-bin-transform");
        m_pEglGles = DSL_ELEMENT_NEW(NVDS_ELEM_SINK_EGL, "sink-bin-eglgles");
        
        m_pEglGles->SetAttribute("window-x", m_offsetX);
        m_pEglGles->SetAttribute("window-y", m_offsetY);
        m_pEglGles->SetAttribute("window-width", m_width);
        m_pEglGles->SetAttribute("window-height", m_height);
        m_pEglGles->SetAttribute("sync", m_sync);
        m_pEglGles->SetAttribute("max-lateness", -1);
        m_pEglGles->SetAttribute("async", m_async);
        m_pEglGles->SetAttribute("qos", m_qos);
        
        AddChild(m_pTransform);
        AddChild(m_pEglGles);
    }
    
    WindowSinkBintr::~WindowSinkBintr()
    {
        LOG_FUNC();
    
        if (IsLinked())
        {    
            UnlinkAll();
        }
    }

    bool WindowSinkBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("OverlaySinkBintr '" << m_name << "' is already linked");
            return false;
        }
        if (!m_pQueue->LinkToSink(m_pTransform) or
            !m_pTransform->LinkToSink(m_pEglGles))
        {
            return false;
        }
        m_isLinked = true;
        return true;
    }
    
    void WindowSinkBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("OverlaySinkBintr '" << m_name << "' is not linked");
            return;
        }
        m_pQueue->UnlinkFromSink();
        m_pTransform->UnlinkFromSink();
        m_isLinked = false;
    }

    void  WindowSinkBintr::GetOffsets(uint* offsetX, uint* offsetY)
    {
        LOG_FUNC();
        
        *offsetX = m_offsetX;
        *offsetY = m_offsetY;
    }
    
    bool WindowSinkBintr::SetOffsets(uint offsetX, uint offsetY)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Dimensions for WindowSinkBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_offsetX = offsetX;
        m_offsetY = offsetY;

        m_pEglGles->SetAttribute("window-x", m_offsetX);
        m_pEglGles->SetAttribute("window-y", m_offsetY);
        
        return true;
    }

    void WindowSinkBintr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_width;
        *height = m_height;
    }

    bool WindowSinkBintr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Dimensions for WindowSinkBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_width = width;
        m_height = height;

        m_pEglGles->SetAttribute("window-width", m_width);
        m_pEglGles->SetAttribute("window-height", m_height);
        
        return true;
    }
    
    //-------------------------------------------------------------------------
    
    FileSinkBintr::FileSinkBintr(const char* name, const char* filepath, 
        uint codec, uint container, uint bitRate, uint interval)
        : SinkBintr(name)
        , m_sync(TRUE)
        , m_async(FALSE)
        , m_codec(codec)
        , m_bitRate(bitRate)
        , m_interval(interval)
        , m_container(container)
    {
        LOG_FUNC();
        
        m_isWindowCapable = false;

        m_pFileSink = DSL_ELEMENT_NEW(NVDS_ELEM_SINK_FILE, "file-sink-bin");
        m_pTransform = DSL_ELEMENT_NEW(NVDS_ELEM_VIDEO_CONV, "file-sink-bin-transform");
        m_pCapsFilter = DSL_ELEMENT_NEW(NVDS_ELEM_CAPS_FILTER, "file-sink-bin-caps-filter");

        m_pFileSink->SetAttribute("location", filepath);
        m_pFileSink->SetAttribute("sync", m_sync);
        m_pFileSink->SetAttribute("async", m_async);
        
        m_pTransform->SetAttribute("gpu-id", m_gpuId);

        GstCaps* pCaps(NULL);
        switch (codec)
        {
        case DSL_CODEC_H264 :
            m_pEncoder = DSL_ELEMENT_NEW(NVDS_ELEM_ENC_H264_HW, "file-sink-bin-encoder");
            m_pEncoder->SetAttribute("bitrate", m_bitRate);
            m_pEncoder->SetAttribute("iframeinterval", m_interval);
            m_pEncoder->SetAttribute("bufapi-version", true);
            m_pParser = DSL_ELEMENT_NEW("h264parse", "file-sink-bin-parser");
            pCaps = gst_caps_from_string("video/x-raw(memory:NVMM), format=I420");
            break;
        case DSL_CODEC_H265 :
            m_pEncoder = DSL_ELEMENT_NEW(NVDS_ELEM_ENC_H265_HW, "file-sink-bin-encoder");
            m_pEncoder->SetAttribute("bitrate", m_bitRate);
            m_pEncoder->SetAttribute("iframeinterval", m_interval);
            m_pEncoder->SetAttribute("bufapi-version", true);
            m_pParser = DSL_ELEMENT_NEW("h265parse", "file-sink-bin-parser");
            pCaps = gst_caps_from_string("video/x-raw(memory:NVMM), format=I420");
            break;
        case DSL_CODEC_MPEG4 :
            m_pEncoder = DSL_ELEMENT_NEW(NVDS_ELEM_ENC_MPEG4, "file-sink-bin-encoder");
            m_pParser = DSL_ELEMENT_NEW("mpeg4videoparse", "file-sink-bin-parser");
            pCaps = gst_caps_from_string("video/x-raw, format=I420");
            break;
        default:
            LOG_ERROR("Invalid codec = '" << codec << "' for new Sink '" << name << "'");
            throw;
        }

        m_pCapsFilter->SetAttribute("caps", pCaps);
        gst_caps_unref(pCaps);
        
        switch (container)
        {
        case DSL_CONTAINER_MP4 :
            m_pContainer = DSL_ELEMENT_NEW(NVDS_ELEM_MUX_MP4, "file-sink-bin-container");        
            break;
        case DSL_CONTAINER_MKV :
            m_pContainer = DSL_ELEMENT_NEW(NVDS_ELEM_MKV, "file-sink-bin-container");        
            break;
        default:
            LOG_ERROR("Invalid container = '" << container << "' for new Sink '" << name << "'");
            throw;
        }

        AddChild(m_pFileSink);
        AddChild(m_pTransform);
        AddChild(m_pCapsFilter);
        AddChild(m_pEncoder);
        AddChild(m_pParser);
        AddChild(m_pContainer);
    }
    
    FileSinkBintr::~FileSinkBintr()
    {
        LOG_FUNC();
    
        if (IsLinked())
        {    
            UnlinkAll();
        }
    }

    bool FileSinkBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("FileSinkBintr '" << m_name << "' is already linked");
            return false;
        }
        if (!m_pQueue->LinkToSink(m_pTransform) or
            !m_pTransform->LinkToSink(m_pCapsFilter) or
            !m_pCapsFilter->LinkToSink(m_pEncoder) or
            !m_pEncoder->LinkToSink(m_pParser) or
            !m_pParser->LinkToSink(m_pContainer) or
            !m_pContainer->LinkToSink(m_pFileSink))
        {
            return false;
        }
        m_isLinked = true;
        return true;
    }
    
    void FileSinkBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("FileSinkBintr '" << m_name << "' is not linked");
            return;
        }
        m_pContainer->UnlinkFromSink();
        m_pParser->UnlinkFromSink();
        m_pEncoder->UnlinkFromSink();
        m_pCapsFilter->UnlinkFromSink();
        m_pTransform->UnlinkFromSink();
        m_pQueue->UnlinkFromSink();
        m_isLinked = false;
    }

    void  FileSinkBintr::GetVideoFormats(uint* codec, uint* container)
    {
        LOG_FUNC();
        
        *codec = m_codec;
        *container = m_container;
    }
    
    void  FileSinkBintr::GetEncoderSettings(uint* bitRate, uint* interval)
    {
        LOG_FUNC();
        
        *bitRate = m_bitRate;
        *interval = m_interval;
    }
    
    bool FileSinkBintr::SetEncoderSettings(uint bitRate, uint interval)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Encoder Settings for FileSinkBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_bitRate = bitRate;
        m_interval = interval;

        if (m_codec == DSL_CODEC_H264 or m_codec == DSL_CODEC_H265)
        {
            m_pEncoder->SetAttribute("bitrate", m_bitRate);
            m_pEncoder->SetAttribute("iframeinterval", m_interval);
        }
        return true;
    }
    
    bool FileSinkBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set GPU ID for FileSinkBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_gpuId = gpuId;
        LOG_DEBUG("Setting GPU ID to '" << gpuId << "' for FileSinkBintr '" << m_name << "'");

        m_pTransform->SetAttribute("gpu-id", m_gpuId);
        
        return true;
    }
    
    //-------------------------------------------------------------------------
    
    RecordSinkBintr::RecordSinkBintr(const char* name, const char* outdir, uint container, 
        NvDsSRCallbackFunc clientListener)
        : SinkBintr(name)
        , m_outdir(outdir)
        , m_pContext(NULL)
    {
        LOG_FUNC();
        
        switch (container)
        {
        case DSL_CONTAINER_MP4 :
            m_initParams.containerType = NVDSSR_CONTAINER_MP4;        
            break;
        case DSL_CONTAINER_MKV :
            m_initParams.containerType = NVDSSR_CONTAINER_MKV;        
            break;
        default:
            LOG_ERROR("Invalid container = '" << container << "' for new RecordSinkBintr '" << name << "'");
            throw;
        }
        
        // Set single callback listener. Unique clients must identifed using client data provided on Start session
        m_initParams.callback = clientListener;
        
        // Set both width and height params to zero = no-transcode
        m_initParams.width = 0;  
        m_initParams.height = 0; 
        
        // Filename prefix uses bintr name by default
        m_initParams.fileNamePrefix = const_cast<gchar*>(GetCStrName());
        m_initParams.dirpath = const_cast<gchar*>(m_outdir.c_str());
        
        m_initParams.defaultDuration = DSL_DEFAULT_SINK_VIDEO_DURATION_IN_SEC;
        m_initParams.videoCacheSize = DSL_DEFAULT_SINK_VIDEO_CACHE_IN_SEC;

    }
    
    RecordSinkBintr::~RecordSinkBintr()
    {
        LOG_FUNC();
    
        if (IsLinked())
        {    
            UnlinkAll();
        }
    }

    bool RecordSinkBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("RecordSinkBintr '" << m_name << "' is already linked");
            return false;
        }

        // Create the smart record context
        if (NvDsSRCreate(&m_pContext, &m_initParams) != NVDSSR_STATUS_OK)
        {
            LOG_ERROR("Failed to create Smart Record Context for new RecordSinkBintr '" << m_name << "'");
            return false;
        }
        
        m_pRecordBin = DSL_NODETR_NEW("record-bin");
        m_pRecordBin->SetGstObject(GST_OBJECT(m_pContext->recordbin));
            
        AddChild(m_pRecordBin);

        if (!m_pQueue->LinkToSink(m_pRecordBin))
        {
            return false;
        }
        m_isLinked = true;
        return true;
    }
    
    void RecordSinkBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("FileSinkBintr '" << m_name << "' is not linked");
            return;
        }
        m_pQueue->UnlinkFromSink();
        RemoveChild(m_pRecordBin);
        
        m_pRecordBin = nullptr;
        NvDsSRDestroy(m_pContext);
        
        m_isLinked = false;
    }

    const char* RecordSinkBintr::GetOutdir()
    {
        LOG_FUNC();
        
        return m_outdir.c_str();
    }
    
    bool RecordSinkBintr::SetOutdir(const char* outdir)
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("Unable to set the Output for RecordSinkBintr '" << GetName() 
                << "' as it's currently Linked");
            return false;
        }
        
        m_outdir.assign(outdir);
        return true;
    }

    uint RecordSinkBintr::GetCacheSize()
    {
        LOG_FUNC();
        
        return m_initParams.videoCacheSize;
    }

    bool RecordSinkBintr::SetCacheSize(uint videoCacheSize)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set cache size for RecordSinkBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_initParams.videoCacheSize = videoCacheSize;
        
        return true;
    }


    void RecordSinkBintr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_initParams.width;
        *height = m_initParams.height;
    }

    bool RecordSinkBintr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Dimensions for RecordSinkBintr '" << GetName() 
                << "' as it's currently Linked");
            return false;
        }

        m_initParams.width = width;
        m_initParams.height = height;
        
        return true;
    }
    
    bool RecordSinkBintr::StartSession(uint* session, uint start, uint duration, void* clientData)
    {
        LOG_FUNC();
        
        if (!IsLinked())
        {
            LOG_ERROR("Unable to Start Session for RecordSinkBintr '" << GetName() 
                << "' as it is not currently Linked");
            return false;
        }
        return (NvDsSRStart(m_pContext, session, start, duration, clientData) == NVDSSR_STATUS_OK);
    }
    
    bool RecordSinkBintr::StopSession(uint session)
    {
        LOG_FUNC();
        
        if (!IsLinked())
        {
            LOG_ERROR("Unable to Stop Session for RecordSinkBintr '" << GetName() 
                << "' as it is not currently Linked");
            return false;
        }
        return (NvDsSRStop(m_pContext, session) == NVDSSR_STATUS_OK);
    }
    

    //******************************************************************************************
    
    RtspSinkBintr::RtspSinkBintr(const char* name, const char* host, uint udpPort, uint rtspPort,
         uint codec, uint bitRate, uint interval)
        : SinkBintr(name)
        , m_host(host)
        , m_udpPort(udpPort)
        , m_rtspPort(rtspPort)
        , m_sync(FALSE)
        , m_async(FALSE)
        , m_codec(codec)
        , m_bitRate(bitRate)
        , m_interval(interval)
        , m_pServer(NULL)
        , m_pFactory(NULL)
    {
        LOG_FUNC();
        
        m_isWindowCapable = false;

        m_pUdpSink = DSL_ELEMENT_NEW("udpsink", "rtsp-sink-bin");
        m_pTransform = DSL_ELEMENT_NEW(NVDS_ELEM_VIDEO_CONV, "rtsp-sink-bin-transform");
        m_pCapsFilter = DSL_ELEMENT_NEW(NVDS_ELEM_CAPS_FILTER, "rtsp-sink-bin-caps-filter");

        m_pUdpSink->SetAttribute("host", m_host.c_str());
        m_pUdpSink->SetAttribute("port", m_udpPort);
        m_pUdpSink->SetAttribute("sync", m_sync);
        m_pUdpSink->SetAttribute("async", m_async);

        GstCaps* pCaps = gst_caps_from_string("video/x-raw(memory:NVMM), format=I420");
        m_pCapsFilter->SetAttribute("caps", pCaps);
        gst_caps_unref(pCaps);
        
        std::string codecString;
        switch (codec)
        {
        case DSL_CODEC_H264 :
            m_pEncoder = DSL_ELEMENT_NEW(NVDS_ELEM_ENC_H264_HW, "rtsp-sink-bin-h264-encoder");
            m_pParser = DSL_ELEMENT_NEW("h264parse", "rtsp-sink-bin-h264-parser");
            m_pPayloader = DSL_ELEMENT_NEW("rtph264pay", "rtsp-sink-bin-h264-payloader");
            codecString.assign("H264");
            break;
        case DSL_CODEC_H265 :
            m_pEncoder = DSL_ELEMENT_NEW(NVDS_ELEM_ENC_H265_HW, "rtsp-sink-bin-h265-encoder");
            m_pParser = DSL_ELEMENT_NEW("h265parse", "rtsp-sink-bin-h265-parser");
            m_pPayloader = DSL_ELEMENT_NEW("rtph265pay", "rtsp-sink-bin-h265-payloader");
            codecString.assign("H265");
            break;
        default:
            LOG_ERROR("Invalid codec = '" << codec << "' for new Sink '" << name << "'");
            throw;
        }

        m_pEncoder->SetAttribute("bitrate", m_bitRate);
        m_pEncoder->SetAttribute("iframeinterval", m_interval);
        m_pEncoder->SetAttribute("preset-level", true);
        m_pEncoder->SetAttribute("insert-sps-pps", true);
        m_pEncoder->SetAttribute("bufapi-version", true);
        
        // Setup the GST RTSP Server
        m_pServer = gst_rtsp_server_new();
        g_object_set(m_pServer, "service", std::to_string(m_rtspPort).c_str(), NULL);

        std::string udpSrc = "(udpsrc name=pay0 port=" + std::to_string(m_udpPort) + 
            " caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=" +
            codecString + ", payload=96 \")";
        
        // Create a nw RTSP Media Factory and set the launch settings
        // to the UDP source defined above
        m_pFactory = gst_rtsp_media_factory_new();
        gst_rtsp_media_factory_set_launch(m_pFactory, udpSrc.c_str());

        LOG_INFO("UDP Src for RtspSinkBintr '" << m_name << "' = " << udpSrc);

        // Get a handle to the Mount-Points object from the new RTSP Server
        GstRTSPMountPoints* pMounts = gst_rtsp_server_get_mount_points(m_pServer);

        // Attach the RTSP Media Factory to the mount-point-path in the mounts object.
        std::string uniquePath = "/" + m_name;
        gst_rtsp_mount_points_add_factory(pMounts, uniquePath.c_str(), m_pFactory);
        g_object_unref(pMounts);

        AddChild(m_pUdpSink);
        AddChild(m_pTransform);
        AddChild(m_pCapsFilter);
        AddChild(m_pEncoder);
        AddChild(m_pParser);
        AddChild(m_pPayloader);
    }
    
    RtspSinkBintr::~RtspSinkBintr()
    {
        LOG_FUNC();
    
        if (IsLinked())
        {    
            UnlinkAll();
        }
    }

    bool RtspSinkBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("RtspSinkBintr '" << m_name << "' is already linked");
            return false;
        }
        
        if (!m_pQueue->LinkToSink(m_pTransform) or
            !m_pTransform->LinkToSink(m_pCapsFilter) or
            !m_pCapsFilter->LinkToSink(m_pEncoder) or
            !m_pEncoder->LinkToSink(m_pParser) or
            !m_pParser->LinkToSink(m_pPayloader) or
            !m_pPayloader->LinkToSink(m_pUdpSink))
        {
            return false;
        }

        // Attach the server to the Main loop context. Server will accept
        // connections the once main loop has been started
        m_pServerSrcId = gst_rtsp_server_attach(m_pServer, NULL);

        m_isLinked = true;
        return true;
    }
    
    void RtspSinkBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("RtspSinkBintr '" << m_name << "' is not linked");
            return;
        }
        if (m_pServerSrcId)
        {
            // Remove (destroy) the source from the Main loop context
            g_source_remove(m_pServerSrcId);
            m_pServerSrcId = 0;
        }
        
        m_pPayloader->UnlinkFromSink();
        m_pParser->UnlinkFromSink();
        m_pEncoder->UnlinkFromSink();
        m_pCapsFilter->UnlinkFromSink();
        m_pTransform->UnlinkFromSink();
        m_pQueue->UnlinkFromSink();
        m_isLinked = false;
    }
    
    void RtspSinkBintr::GetServerSettings(uint* udpPort, uint* rtspPort, uint* codec)
    {
        LOG_FUNC();
        
        *udpPort = m_udpPort;
        *rtspPort = m_rtspPort;
        *codec = m_codec;
    }
    
    void  RtspSinkBintr::GetEncoderSettings(uint* bitRate, uint* interval)
    {
        LOG_FUNC();
        
        *bitRate = m_bitRate;
        *interval = m_interval;
    }
    
    bool RtspSinkBintr::SetEncoderSettings(uint bitRate, uint interval)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Encoder Settings for FileSinkBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_bitRate = bitRate;
        m_interval = interval;

        if (m_codec == DSL_CODEC_H264 or m_codec == DSL_CODEC_H265)
        {
            m_pEncoder->SetAttribute("bitrate", m_bitRate);
            m_pEncoder->SetAttribute("iframeinterval", m_interval);
        }
        return true;
    }
    
    //-------------------------------------------------------------------------

    ImageSinkBintr::ImageSinkBintr(const char* name, const char* outdir)
        : FakeSinkBintr(name)
        , m_outdir(outdir)
        , m_frameCaptureframeCount(0)
        , m_frameCaptureInterval(0)
        , m_isFrameCaptureEnabled(false)
        , m_objectCaptureFrameCount(0)
        , m_isObjectCaptureEnabled(false)
    {
        LOG_FUNC();
        
        g_mutex_init(&m_captureMutex);
        
        m_pSinkPadProbe = DSL_PAD_PROBE_NEW("image-sink-pad-probe", "sink", m_pQueue);
    }
    
    ImageSinkBintr::~ImageSinkBintr()
    {
        LOG_FUNC();
        
        g_mutex_clear(&m_captureMutex);
    }

    const char* ImageSinkBintr::GetOutdir()
    {
        LOG_FUNC();
        
        return m_outdir.c_str();
    }
    
    bool ImageSinkBintr::SetOutdir(const char* outdir)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureMutex);
        
        m_outdir.assign(outdir);
        return true;
    }

    uint ImageSinkBintr::GetFrameCaptureInterval()
    {
        LOG_FUNC();
        
        return m_frameCaptureInterval;
    }

    bool ImageSinkBintr::SetFrameCaptureInterval(uint frameCaptureInterval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureMutex);
        
        m_frameCaptureInterval = frameCaptureInterval;
        return true;
    }

    bool ImageSinkBintr::GetFrameCaptureEnabled()
    {
        LOG_FUNC();
        
        return m_isFrameCaptureEnabled;
    }
    
    bool ImageSinkBintr::SetFrameCaptureEnabled(bool enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureMutex);

        if (m_isFrameCaptureEnabled == enabled)
        {
            LOG_ERROR("Can't set Frame Capture Enabled to the same value of " 
                << enabled << " for ImageSinkBintr '" << GetName() << "' ");
            return false;
        }
        m_isFrameCaptureEnabled = enabled;
        
        if (enabled)
        {
            LOG_INFO("Enabling Frame Capture for ImageSinkBintr '" << GetName() << "'");
            
            // reset the Frame count for new capture
            m_frameCaptureframeCount = 0;
            return AddBatchMetaHandler(DSL_PAD_SINK, FrameCaptureHandler, this);
        }
        LOG_INFO("Disabling Frame Capture for ImageSinkBintr '" << GetName() << "'");
        
        return RemoveBatchMetaHandler(DSL_PAD_SINK, FrameCaptureHandler);
    }
    
    bool ImageSinkBintr::GetObjectCaptureEnabled()
    {
        LOG_FUNC();
        
        return m_isObjectCaptureEnabled;
    }
    
    bool ImageSinkBintr::SetObjectCaptureEnabled(bool enabled)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureMutex);

        if (m_isObjectCaptureEnabled == enabled)
        {
            LOG_ERROR("Can't set Object Capture Enabled to the same value of " 
                << enabled << " for ImageSinkBintr '" << GetName() << "' ");
            return false;
        }
        m_isObjectCaptureEnabled = enabled;
        
        if (enabled)
        {
            LOG_INFO("Enabling Object Capture for ImageSinkBintr '" << GetName() << "'");
            
            // reset the Frame count for new capture
            m_objectCaptureFrameCount = 0;
            return AddBatchMetaHandler(DSL_PAD_SINK, ObjectCaptureHandler, this);
        }
        LOG_INFO("Disabling Object Capture for ImageSinkBintr '" << GetName() << "'");
        
        return RemoveBatchMetaHandler(DSL_PAD_SINK, ObjectCaptureHandler);
    }
    
    bool ImageSinkBintr::AddObjectCaptureClass(uint classId, boolean fullFrame, uint captureLimit)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureMutex);

        if (m_captureClasses.find(classId) != m_captureClasses.end())
        {
            LOG_ERROR("ImageSinkBintr '" << GetName() <<"' has an existing Capture Class with ID " << classId);
            return false;
        }
        LOG_INFO("Adding Object Capture Class " << classId << " for ImageSinkBintr '" << GetName() << "'");

        std::shared_ptr<CaptureClass> pCaptureClass = 
            std::shared_ptr<CaptureClass>(new CaptureClass(classId, fullFrame, captureLimit));

        m_captureClasses[classId] = pCaptureClass;
        return true;
    }
    
    bool ImageSinkBintr::RemoveObjectCaptureClass(uint classId)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureMutex);
        
        if (m_captureClasses.find(classId) == m_captureClasses.end())
        {
            LOG_ERROR("ImageSinkBintr '" << GetName() <<"' does not have Capture Class with ID " << classId);
            return false;
        }
        LOG_INFO("Removing Object Capture Class " << classId << " for ImageSinkBintr '" << GetName() << "'");

        m_captureClasses.erase(classId);
        return true;
    }

    static bool TransformAndSave(NvBufSurface* surface, const std::string& filespec, 
        NvBufSurfTransformRect& src_rect, NvBufSurfTransformRect& dst_rect)
    {
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
    }

    bool ImageSinkBintr::HandleFrameCapture(GstBuffer* pBuffer)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureMutex);

        if (++m_frameCaptureframeCount % (m_frameCaptureInterval+1))
        {
            return GST_PAD_PROBE_OK;
        }
            
        GstMapInfo inMapInfo = {0};

        if (!gst_buffer_map(pBuffer, &inMapInfo, GST_MAP_READ))
        {
            LOG_ERROR("ImageSinkBintr '" << GetName() << "' failed to map gst buffer");
            gst_buffer_unmap(pBuffer, &inMapInfo);
            return GST_PAD_PROBE_OK;
        }
        
        NvBufSurface* surface = (NvBufSurface*)inMapInfo.data;  
        
        LOG_INFO("transforming frame surface with width "<< surface->surfaceList[0].width 
            << " and height "<< surface->surfaceList[0].height);

        std::string filespec = m_outdir + "/frame" + std::to_string(m_frameCaptureframeCount) + ".jpeg";

        NvBufSurfTransformRect srcRect = {0, 0, surface->surfaceList[0].width, surface->surfaceList[0].height};
        NvBufSurfTransformRect dstRect = {0, 0, surface->surfaceList[0].width, surface->surfaceList[0].height};

        bool retVal = TransformAndSave(surface, filespec, srcRect, dstRect);

        gst_buffer_unmap(pBuffer, &inMapInfo);
        
        return retVal;
    }
    
    bool ImageSinkBintr::HandleObjectCapture(GstBuffer* pBuffer)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_captureMutex);

        m_objectCaptureFrameCount++;
        GstMapInfo inMapInfo = {0};

        if (!gst_buffer_map(pBuffer, &inMapInfo, GST_MAP_READ))
        {
            LOG_ERROR("ImageSinkBintr '" << GetName() << "' failed to map gst buffer");
            gst_buffer_unmap(pBuffer, &inMapInfo);
            return GST_PAD_PROBE_OK;
        }
        NvBufSurface* surface = (NvBufSurface*)inMapInfo.data;  
        NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(pBuffer);
        
        // Iterate through the list of frames to access the meta data for each object
        
        for (NvDsMetaList* l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
        {
            NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);

            if (frame_meta == NULL)
            {
                LOG_DEBUG("NvDS Meta contained NULL frame_meta for ImageSinkBintr '" << GetName() << "'");
                return true;
            }
            // unique object id, per object per frame
            uint objectId(0);
            for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
            {
        
                NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) (l_obj->data);

                NvOSD_RectParams * rect_params = &(obj_meta->rect_params);

                // if the object's classId is enabled for capture 
                if (m_captureClasses.find(obj_meta->class_id) != m_captureClasses.end())
                {
                    // ensue that we don't exceed the maximun number of captures for this class
                    if (m_captureClasses[obj_meta->class_id]->m_captureLimit == 0 or
                        m_captureClasses[obj_meta->class_id]->m_captureCount < 
                        m_captureClasses[obj_meta->class_id]->m_captureLimit)
                    {
                        m_captureClasses[obj_meta->class_id]->m_captureCount++;
                            
                        LOG_INFO("transforming frame surface for classId " << obj_meta->class_id << " with width "
                            << rect_params->width << " and height "<< rect_params->height);

                        std::string filespec = m_outdir + "/frame_" + std::to_string(m_objectCaptureFrameCount) + 
                            "_class_" + std::to_string(obj_meta->class_id) + "_object_" + std::to_string(++objectId) + ".jpeg";
                        
                        // capturing full frame or bbox rectangle only?
                        if (m_captureClasses[obj_meta->class_id]->m_fullFrame)
                        {
                            NvBufSurfTransformRect srcRect = {0, 0, (uint)surface->surfaceList[0].width, surface->surfaceList[0].height};
                            NvBufSurfTransformRect dstRect = {0, 0, (uint)surface->surfaceList[0].width, surface->surfaceList[0].height};
                            TransformAndSave(surface, filespec, srcRect, dstRect);
                        }
                        else
                        {
                            NvBufSurfTransformRect srcRect = {(uint)rect_params->top, (uint)rect_params->left, 
                                (uint)rect_params->width, (uint)rect_params->height};
                            NvBufSurfTransformRect dstRect = {0, 0, (uint)rect_params->width, (uint)rect_params->height};
                            TransformAndSave(surface, filespec, srcRect, dstRect);
                        }
                    }
                }
            }
        }
        gst_buffer_unmap(pBuffer, &inMapInfo);
        
        return true;
    }
    
    static boolean FrameCaptureHandler(void* batch_meta, void* user_data)
    {
        return static_cast<ImageSinkBintr*>(user_data)->
            HandleFrameCapture((GstBuffer*)batch_meta);
    }
    
    static boolean ObjectCaptureHandler(void* batch_meta, void* user_data)
    {
        return static_cast<ImageSinkBintr*>(user_data)->
            HandleObjectCapture((GstBuffer*)batch_meta);
    }
    
}    