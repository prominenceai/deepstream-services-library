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
#include "DslSourceBintr.h"
#include "DslPipelineBintr.h"
#include <nvdsgstutils.h>

#define N_DECODE_SURFACES 16
#define N_EXTRA_SURFACES 1

namespace DSL
{
    SourceBintr::SourceBintr(const char* name)
        : Bintr(name)
        , m_isLive(TRUE)
        , m_width(0)
        , m_height(0)
        , m_fps_n(0)
        , m_fps_d(0)
        , m_latency(100)
        , m_numDecodeSurfaces(N_DECODE_SURFACES)
        , m_numExtraSurfaces(N_EXTRA_SURFACES)
    {
        LOG_FUNC();
    }
    
    SourceBintr::~SourceBintr()
    {
        LOG_FUNC();
    }
    
    bool SourceBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' Source to the Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            AddSourceBintr(std::dynamic_pointer_cast<SourceBintr>(shared_from_this()));
    }

    bool SourceBintr::IsParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // check if 'this' Source is child of Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            IsSourceBintrChild(std::dynamic_pointer_cast<SourceBintr>(shared_from_this()));
    }

    bool SourceBintr::RemoveFromParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        if (!IsParent(pParentBintr))
        {
            LOG_ERROR("Source '" << GetName() << "' is not a child of Pipeline '" << pParentBintr->GetName() << "'");
            return false;
        }
        
        // remove 'this' Source from the Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            RemoveSourceBintr(std::dynamic_pointer_cast<SourceBintr>(shared_from_this()));
    }

    void SourceBintr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_width;
        *height = m_height;
    }

    void  SourceBintr::GetFrameRate(uint* fps_n, uint* fps_d)
    {
        LOG_FUNC();
        
        *fps_n = m_fps_n;
        *fps_d = m_fps_d;
    }

    bool SourceBintr::LinkToSink(DSL_NODETR_PTR pStreamMux) 
    {
        LOG_FUNC();

        std::string sinkPadName = "sink_" + std::to_string(m_uniqueId);
        
        LOG_INFO("Linking Source '" << GetName() << "' to Pad '" << sinkPadName 
            << "' for StreamMux '" << pStreamMux->GetName() << "'");
       
        m_pGstStaticSourcePad = gst_element_get_static_pad(GetGstElement(), "src");
        if (!m_pGstStaticSourcePad)
        {
            LOG_ERROR("Failed to get Static Source Pad for Streaming Source '" << GetName() << "'");
            return false;
        }

        GstPad* pGstRequestedSinkPad = gst_element_get_request_pad(pStreamMux->GetGstElement(), sinkPadName.c_str());
            
        if (!pGstRequestedSinkPad)
        {
            LOG_ERROR("Failed to get Requested Sink Pad for StreamMux '" << pStreamMux->GetName() << "'");
            return false;
        }
        m_pGstRequestedSinkPads[sinkPadName] = pGstRequestedSinkPad;
            
        return Bintr::LinkToSink(pStreamMux);
    }

    bool SourceBintr::UnlinkFromSink()
    {
        LOG_FUNC();

        // If we're currently linked to the StreamMuxer
        if (!IsLinkedToSink())
        {
            LOG_ERROR("SourceBintr '" << GetName() << "' is not in a Linked state");
            return false;
        }

        std::string sinkPadName = "sink_" + std::to_string(m_uniqueId);

        LOG_INFO("Unlinking and releasing request Sink Pad for StreamMux " << m_pSink->GetName());

        gst_pad_send_event(m_pGstRequestedSinkPads[sinkPadName], gst_event_new_flush_stop(FALSE));
        if (!gst_pad_unlink(m_pGstStaticSourcePad, m_pGstRequestedSinkPads[sinkPadName]))
        {
            LOG_ERROR("SourceBintr '" << GetName() << "' failed to unlink from StreamMuxer");
            return false;
        }
        gst_element_release_request_pad(GetSink()->GetGstElement(), m_pGstRequestedSinkPads[sinkPadName]);
        gst_object_unref(m_pGstRequestedSinkPads[sinkPadName]);

        m_pGstRequestedSinkPads.erase(sinkPadName);
        return Bintr::UnlinkFromSink();
    }
    
    //*********************************************************************************

    CsiSourceBintr::CsiSourceBintr(const char* name, 
        guint width, guint height, guint fps_n, guint fps_d)
        : SourceBintr(name)
        , m_sensorId(0)
    {
        LOG_FUNC();

        m_width = width;
        m_height = height;
        m_fps_n = fps_n;
        m_fps_d = fps_d;
        
        m_pSourceElement = DSL_ELEMENT_NEW(NVDS_ELEM_SRC_CAMERA_CSI, "csi_camera_elem");
        m_pCapsFilter = DSL_ELEMENT_NEW(NVDS_ELEM_CAPS_FILTER, "src_caps_filter");

        m_pSourceElement->SetAttribute("sensor-id", m_sensorId);
        m_pSourceElement->SetAttribute("bufapi-version", TRUE);
        
        // Note: not present in Deepstream 5.0
        // m_pSourceElement->SetAttribute("maxperf", TRUE);

        GstCaps * pCaps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12",
            "width", G_TYPE_INT, m_width, "height", G_TYPE_INT, m_height, 
            "framerate", GST_TYPE_FRACTION, m_fps_n, m_fps_d, NULL);
        if (!pCaps)
        {
            LOG_ERROR("Failed to create new Simple Capabilities for '" << name << "'");
            throw;  
        }

        GstCapsFeatures *feature = NULL;
        feature = gst_caps_features_new("memory:NVMM", NULL);
        gst_caps_set_features(pCaps, 0, feature);

        m_pCapsFilter->SetAttribute("caps", pCaps);
        
        gst_caps_unref(pCaps);        

        AddChild(m_pSourceElement);
        AddChild(m_pCapsFilter);
        
        m_pCapsFilter->AddGhostPadToParent("src");
    }

    CsiSourceBintr::~CsiSourceBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }
    
    bool CsiSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("CsiSourceBintr '" << GetName() << "' is already in a linked state");
            return false;
        }
        m_pSourceElement->LinkToSink(m_pCapsFilter);
        m_isLinked = true;
        
        return true;
    }

    void CsiSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("CsiSourceBintr '" << GetName() << "' is not in a linked state");
            return;
        }
        m_pSourceElement->UnlinkFromSink();
        m_isLinked = false;
    }

    //*********************************************************************************

    UsbSourceBintr::UsbSourceBintr(const char* name, 
        guint width, guint height, guint fps_n, guint fps_d)
        : SourceBintr(name)
        , m_sensorId(0)
    {
        LOG_FUNC();

        m_width = width;
        m_height = height;
        m_fps_n = fps_n;
        m_fps_d = fps_d;
        
        m_pSourceElement = DSL_ELEMENT_NEW(NVDS_ELEM_SRC_CAMERA_V4L2, "usb_camera_elem");
        m_pCapsFilter = DSL_ELEMENT_NEW(NVDS_ELEM_CAPS_FILTER, "src_caps_filter");
        m_pVidConv1 = DSL_ELEMENT_NEW(NVDS_ELEM_VIDEO_CONV, "src_video_conv1");
        m_pVidConv2 = DSL_ELEMENT_NEW(NVDS_ELEM_VIDEO_CONV, "src_video_conv2");

        GstCaps * pCaps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12",
            "width", G_TYPE_INT, m_width, "height", G_TYPE_INT, m_height, 
            "framerate", GST_TYPE_FRACTION, m_fps_n, m_fps_d, NULL);
        if (!pCaps)
        {
            LOG_ERROR("Failed to create new Simple Capabilities for '" << name << "'");
            throw;  
        }

        GstCapsFeatures *feature = NULL;
        feature = gst_caps_features_new("memory:NVMM", NULL);
        gst_caps_set_features(pCaps, 0, feature);

        m_pCapsFilter->SetAttribute("caps", pCaps);
        
        gst_caps_unref(pCaps);        
        
        m_pVidConv2->SetAttribute("gpu-id", m_gpuId);
        m_pVidConv2->SetAttribute("nvbuf-memory-type", m_nvbufMemoryType);

        AddChild(m_pSourceElement);
        AddChild(m_pCapsFilter);
        AddChild(m_pVidConv1);
        AddChild(m_pVidConv2);
        
        m_pCapsFilter->AddGhostPadToParent("src");
    }

    UsbSourceBintr::~UsbSourceBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }

    bool UsbSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("UsbSourceBintr '" << GetName() << "' is already in a linked state");
            return false;
        }
        if (!m_pSourceElement->LinkToSink(m_pVidConv1) or 
            !m_pVidConv1->LinkToSink(m_pVidConv2) or
            !m_pVidConv2->LinkToSink(m_pCapsFilter))
        {
            return false;
        }
        m_isLinked = true;
        
        return true;
    }

    void UsbSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("UsbSourceBintr '" << GetName() << "' is not in a linked state");
            return;
        }
        m_pVidConv2->UnlinkFromSink();
        m_pVidConv1->UnlinkFromSink();
        m_pSourceElement->UnlinkFromSink();
        m_isLinked = false;
    }
    
    bool UsbSourceBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set GPU ID for UsbSourceBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_gpuId = gpuId;
        LOG_DEBUG("Setting GPU ID to '" << gpuId << "' for UsbSourceBintr '" << m_name << "'");

        m_pVidConv2->SetAttribute("gpu-id", m_gpuId);
        
        return true;
    }

    //*********************************************************************************

    DecodeSourceBintr::DecodeSourceBintr(const char* name, const char* factoryName, const char* uri,
        bool isLive, uint cudadecMemType, uint intraDecode, uint dropFrameInterval)
        : SourceBintr(name)
        , m_cudadecMemtype(cudadecMemType)
        , m_intraDecode(intraDecode)
        , m_dropFrameInterval(dropFrameInterval)
        , m_accumulatedBase(0)
        , m_prevAccumulatedBase(0)
    {
        LOG_FUNC();
        
        m_isLive = isLive;
        m_uri = uri;
        
        // if not a URL
        if ((m_uri.find("http") == std::string::npos) and (m_uri.find("rtsp") == std::string::npos))
        {
            if (isLive)
            {
                LOG_ERROR("Invalid URI '" << uri << "' for Live source '" << name << "'");
                throw;
            }
            std::ifstream streamUriFile(uri);
            if (!streamUriFile.good())
            {
                LOG_ERROR("URI Source'" << uri << "' Not found");
                throw;
            }
            // File source, not live - setup full path
            char absolutePath[PATH_MAX+1];
            m_uri.assign(realpath(uri, absolutePath));
            m_uri.insert(0, "file:");
        }
        
        LOG_INFO("URI Path = " << m_uri);
        std::string sourceElementName = "src-element" + GetName();
        m_pSourceElement = DSL_ELEMENT_NEW(factoryName, sourceElementName.c_str());
        
        if (m_uri.find("rtsp") != std::string::npos)
        {
            // Configure the source to generate NTP sync values
            configure_source_for_ntp_sync(m_pSourceElement->GetGstElement());
        }
        
        // Add all new Elementrs as Children to the SourceBintr
        AddChild(m_pSourceElement);
    }
    
    void DecodeSourceBintr::HandleOnChildAdded(GstChildProxy* pChildProxy, GObject* pObject,
        gchar* name)
    {
        LOG_FUNC();
        
        std::string strName = name;

        LOG_DEBUG("Child object with name '" << strName << "'");
        
        if (strName.find("decodebin") != std::string::npos)
        {
            g_signal_connect(G_OBJECT(pObject), "child-added",
                G_CALLBACK(OnChildAddedCB), this);
        }

        else if (strName.find("nvcuvid") != std::string::npos)
        {
            g_object_set(pObject, "gpu-id", m_gpuId, NULL);
            g_object_set(pObject, "cuda-memory-type", m_cudadecMemtype, NULL);
            g_object_set(pObject, "source-id", m_uniqueId, NULL);
            g_object_set(pObject, "num-decode-surfaces", m_numDecodeSurfaces, NULL);
            
            if (m_intraDecode)
            {
                g_object_set(pObject, "Intra-decode", m_intraDecode, NULL);
            }
        }

        else if ((strName.find("omx") != std::string::npos))
        {
            if (m_intraDecode)
            {
                g_object_set(pObject, "skip-frames", 2, NULL);
            }
            g_object_set(pObject, "disable-dvfs", TRUE, NULL);
        }

        else if (strName.find("nvjpegdec") != std::string::npos)
        {
            g_object_set(pObject, "DeepStream", TRUE, NULL);
        }

        else if ((strName.find("nvv4l2decoder") != std::string::npos))
        {
            if (m_intraDecode)
            {
                g_object_set(pObject, "skip-frames", 2, NULL);
            }
            g_object_set(pObject, "enable-max-performance", TRUE, NULL);
            g_object_set(pObject, "bufapi-version", TRUE, NULL);
            g_object_set(pObject, "drop-frame-interval", m_dropFrameInterval, NULL);
            g_object_set(pObject, "num-extra-surfaces", m_numExtraSurfaces, NULL);

            // if the source is from file, then setup Stream buffer probe function
            // to handle the stream restart/loop on GST_EVENT_EOS.
            if (!m_isLive and false)
            {
                GstPadProbeType mask = (GstPadProbeType) 
                    (GST_PAD_PROBE_TYPE_EVENT_BOTH |
                    GST_PAD_PROBE_TYPE_EVENT_FLUSH | 
                    GST_PAD_PROBE_TYPE_BUFFER);
                    
                GstPad* pStaticSinkpad = gst_element_get_static_pad(GST_ELEMENT(pObject), "sink");
                
                m_bufferProbeId = 
                    gst_pad_add_probe(pStaticSinkpad, mask, StreamBufferRestartProbCB, this, NULL);
            }
        }
    }
    
    GstPadProbeReturn DecodeSourceBintr::HandleStreamBufferRestart(GstPad* pPad, GstPadProbeInfo* pInfo)
    {
        LOG_FUNC();

        GstEvent* event = GST_EVENT(pInfo->data);

        if (pInfo->type & GST_PAD_PROBE_TYPE_BUFFER)
        {
            GST_BUFFER_PTS(GST_BUFFER(pInfo->data)) += m_prevAccumulatedBase;
        }
        
        if (pInfo->type & GST_PAD_PROBE_TYPE_EVENT_BOTH)
        {
            if (GST_EVENT_TYPE(event) == GST_EVENT_EOS)
            {
                g_timeout_add(1, StreamBufferSeekCB, this);
            }
            if (GST_EVENT_TYPE(event) == GST_EVENT_SEGMENT)
            {
                GstSegment* segment;

                gst_event_parse_segment(event, (const GstSegment**)&segment);
                segment->base = m_accumulatedBase;
                m_prevAccumulatedBase = m_accumulatedBase;
                m_accumulatedBase += segment->stop;
            }
            switch (GST_EVENT_TYPE (event))
            {
            case GST_EVENT_EOS:
            // QOS events from downstream sink elements cause decoder to drop
            // frames after looping the file since the timestamps reset to 0.
            // We should drop the QOS events since we have custom logic for
            // looping individual sources.
            case GST_EVENT_QOS:
            case GST_EVENT_SEGMENT:
            case GST_EVENT_FLUSH_START:
            case GST_EVENT_FLUSH_STOP:
                return GST_PAD_PROBE_DROP;
            default:
                break;
            }
        }
        return GST_PAD_PROBE_OK;
    }

    void DecodeSourceBintr::HandleOnSourceSetup(GstElement* pObject, GstElement* arg0)
    {
        if (g_object_class_find_property(G_OBJECT_GET_CLASS(arg0), "latency")) 
        {
            g_object_set(G_OBJECT(arg0), "latency", "cb_sourcesetup set %d latency\n", NULL);
        }
    }
    
    gboolean DecodeSourceBintr::HandleStreamBufferSeek()
    {
        SetState(GST_STATE_PAUSED);
        
        gboolean retval = gst_element_seek(GetGstElement(), 1.0, GST_FORMAT_TIME,
            (GstSeekFlags)(GST_SEEK_FLAG_KEY_UNIT | GST_SEEK_FLAG_FLUSH),
            GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_NONE, GST_CLOCK_TIME_NONE);

        if (!retval)
        {
            LOG_WARN("Failure to seek");
        }

        SetState(GST_STATE_PLAYING);
        return false;
    }

    
    bool DecodeSourceBintr::AddDewarperBintr(DSL_BASE_PTR pDewarperBintr)
    {
        LOG_FUNC();
        
        if (m_pDewarperBintr)
        {
            LOG_ERROR("Source '" << GetName() << "' allready has a Dewarper");
            return false;
        }
        m_pDewarperBintr = std::dynamic_pointer_cast<DewarperBintr>(pDewarperBintr);
        AddChild(pDewarperBintr);
        return true;
    }

    bool DecodeSourceBintr::RemoveDewarperBintr()
    {
        LOG_FUNC();

        if (!m_pDewarperBintr)
        {
            LOG_ERROR("Source '" << GetName() << "' does not have a Dewarper");
            return false;
        }
        RemoveChild(m_pDewarperBintr);
        m_pDewarperBintr = nullptr;
        return true;
    }
    
    bool DecodeSourceBintr::HasDewarperBintr()
    {
        LOG_FUNC();
        
        return (m_pDewarperBintr != nullptr);
    }

    //*********************************************************************************

    UriSourceBintr::UriSourceBintr(const char* name, const char* uri, bool isLive,
        uint cudadecMemType, uint intraDecode, uint dropFrameInterval)
        : DecodeSourceBintr(name, NVDS_ELEM_SRC_URI, uri, isLive, cudadecMemType, intraDecode, dropFrameInterval)
    {
        LOG_FUNC();
        
        // New Elementrs for this Source
        m_pSourceQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "src-queue");
        m_pTee = DSL_ELEMENT_NEW(NVDS_ELEM_TEE, "tee");
        m_pFakeSinkQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "fake-sink-queue");
        m_pFakeSink = DSL_ELEMENT_NEW(NVDS_ELEM_SINK_FAKESINK, "fake-sink");

        // Set the URI for Source Elementr
        m_pSourceElement->SetAttribute("uri", m_uri.c_str());

        // Connect UIR Source Setup Callbacks
        g_signal_connect(m_pSourceElement->GetGObject(), "pad-added", 
            G_CALLBACK(UriSourceElementOnPadAddedCB), this);
        g_signal_connect(m_pSourceElement->GetGObject(), "child-added", 
            G_CALLBACK(OnChildAddedCB), this);
        g_object_set_data(G_OBJECT(m_pSourceElement->GetGObject()), "source", this);

        g_signal_connect(m_pSourceElement->GetGObject(), "source-setup",
            G_CALLBACK(OnSourceSetupCB), this);

        m_pFakeSink->SetAttribute("sync", false);
        m_pFakeSink->SetAttribute("async", false);

        // Add all new Elementrs as Children to the SourceBintr
        AddChild(m_pSourceQueue);
        AddChild(m_pTee);
        AddChild(m_pFakeSinkQueue);
        AddChild(m_pFakeSink);
        
        // Source Ghost Pad for Source Queue
        m_pSourceQueue->AddGhostPadToParent("src");
    }

    UriSourceBintr::~UriSourceBintr()
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            UnlinkAll();
        }
    }

    bool UriSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("UriSourceBintr '" << GetName() << "' is already in a linked state");
            return false;
        }

        GstPadTemplate* pPadTemplate = 
            gst_element_class_get_pad_template(GST_ELEMENT_GET_CLASS(m_pTee->GetGstElement()), "src_%u");
        if (!pPadTemplate)
        {
            LOG_ERROR("Failed to get Pad Template for '" << GetName() << "'");
            return false;
        }
        
        // The TEE for this source is linked to both the "source queue" and "fake sink queue"

        GstPad* pGstRequestedSourcePad = gst_element_request_pad(m_pTee->GetGstElement(), pPadTemplate, NULL, NULL);
        if (!pGstRequestedSourcePad)
        {
            LOG_ERROR("Failed to get Tee Pad for PipelineSinksBintr '" << GetName() <<"'");
            return false;
        }
        std::string padForSourceQueueName = "padForSourceQueue_" + std::to_string(m_uniqueId);

        m_pGstRequestedSourcePads[padForSourceQueueName] = pGstRequestedSourcePad;
        
        if (HasDewarperBintr())
        {
            if (!m_pDewarperBintr->LinkToSource(m_pTee) or !m_pDewarperBintr->LinkToSink(m_pSourceQueue))
            {
                return false;
            }            
        }
        else
        {
            if (!m_pSourceQueue->LinkToSource(m_pTee))
            {
                return false;
            }
        }

        pGstRequestedSourcePad = gst_element_request_pad(m_pTee->GetGstElement(), pPadTemplate, NULL, NULL);
        if (!pGstRequestedSourcePad)
        {
            LOG_ERROR("Failed to get Tee Pad for PipelineSinksBintr '" << GetName() <<"'");
            return false;
        }
        std::string padForFakeSinkQueueName = "padForFakeSinkQueue_" + std::to_string(m_uniqueId);

        m_pGstRequestedSourcePads[padForFakeSinkQueueName] = pGstRequestedSourcePad;

        if (!m_pFakeSinkQueue->LinkToSource(m_pTee) or !m_pFakeSinkQueue->LinkToSink(m_pFakeSink))
        {
            return false;
        }
        m_isLinked = true;

        return true;
    }

    void UriSourceBintr::UnlinkAll()
    {
        LOG_FUNC();
    
        if (!m_isLinked)
        {
            LOG_ERROR("UriSourceBintr '" << GetName() << "' is not in a linked state");
            return;
        }
        m_pFakeSinkQueue->UnlinkFromSource();
        m_pFakeSinkQueue->UnlinkFromSink();

        if (HasDewarperBintr())
        {
            m_pDewarperBintr->UnlinkFromSource();
            m_pDewarperBintr->UnlinkFromSink();
        }
        else
        {
            m_pSourceQueue->UnlinkFromSource();
        }

        for (auto const& imap: m_pGstRequestedSourcePads)
        {
            gst_element_release_request_pad(m_pTee->GetGstElement(), imap.second);
            gst_object_unref(imap.second);
        }
        
        m_isLinked = false;
    }

    void UriSourceBintr::HandleSourceElementOnPadAdded(GstElement* pBin, GstPad* pPad)
    {
        LOG_FUNC();

        // The "pad-added" callback will be called twice for each URI source,
        // once each for the decoded Audio and Video streams. Since we only 
        // want to link to the Video source pad, we need to know which of the
        // two streams this call is for.
        GstCaps* pCaps = gst_pad_query_caps(pPad, NULL);
        GstStructure* structure = gst_caps_get_structure(pCaps, 0);
        std::string name = gst_structure_get_name(structure);
        
        LOG_INFO("Caps structs name " << name);
        if (name.find("video") != std::string::npos)
        {
            m_pGstStaticSinkPad = gst_element_get_static_pad(m_pTee->GetGstElement(), "sink");
            if (!m_pGstStaticSinkPad)
            {
                LOG_ERROR("Failed to get Static Source Pad for Streaming Source '" << GetName() << "'");
            }
            
            if (gst_pad_link(pPad, m_pGstStaticSinkPad) != GST_PAD_LINK_OK) 
            {
                LOG_ERROR("Failed to link decodebin to pipeline");
                throw;
            }
            
            // Update the cap memebers for this URI Source Bintr
            gst_structure_get_uint(structure, "width", &m_width);
            gst_structure_get_uint(structure, "height", &m_height);
            gst_structure_get_fraction(structure, "framerate", (gint*)&m_fps_n, (gint*)&m_fps_d);
            
            LOG_INFO("Video decode linked for URI source '" << GetName() << "'");
        }
    }

    bool UriSourceBintr::SetUri(const char* uri)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Uri for UriSourceBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }
        std::string newUri(uri);
        if (newUri.find("http") == std::string::npos)
        {
            if (m_isLive)
            {
                LOG_ERROR("Invalid URI '" << uri << "' for Live source '" << GetName() << "'");
                return false;
            }
            std::ifstream streamUriFile(uri);
            if (!streamUriFile.good())
            {
                LOG_ERROR("URI Source'" << uri << "' Not found");
                return false;
            }
            // File source, not live - setup full path
            char absolutePath[PATH_MAX+1];
            m_uri.assign(realpath(uri, absolutePath));
            m_uri.insert(0, "file:");
        }        
        m_pSourceElement->SetAttribute("uri", m_uri.c_str());
        
        return true;
    }
    
    //*********************************************************************************
    
    RtspSourceBintr::RtspSourceBintr(const char* name, const char* uri, uint protocol,
        uint cudadecMemType, uint intraDecode, uint dropFrameInterval, uint latency, uint timeout)
        : DecodeSourceBintr(name, "rtspsrc", uri, true, cudadecMemType, intraDecode, dropFrameInterval)
        , m_rtpProtocols(protocol)
        , m_bufferTimeout(timeout)
        , m_streamMgtTimerId(0)
        , m_lastReconnectTime{0}
        , m_lastReconnectCount(0)
        , m_resetMgtTimerId(0)
        , m_isInReset(false)
        , m_lastResetTime{0}
        , m_lastResetCount(0)
    {
        LOG_FUNC();
        
        // Set RTSP latency
        m_latency = latency;
        LOG_DEBUG("Setting latency to '" << latency << "' for RtspSourceBintr '" << m_name << "'");

        // New RTSP Specific Elementrs for this Source
        m_pPreDecodeTee = DSL_ELEMENT_NEW(NVDS_ELEM_TEE, "pre-decode-tee");
        m_pPreDecodeQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "pre-decode-queue");
        m_pDecodeBin = DSL_ELEMENT_NEW("decodebin", "decode-bin");
        m_pSourceQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "src-queue");

        m_pSourceElement->SetAttribute("location", m_uri.c_str());
        m_pSourceElement->SetAttribute("latency", m_latency);
        m_pSourceElement->SetAttribute("drop-on-latency", true);
        m_pSourceElement->SetAttribute("protocols", m_rtpProtocols);

        g_signal_connect (m_pSourceElement->GetGObject(), "select-stream",
            G_CALLBACK(RtspSourceSelectStreamCB), this);

        // Connect RTSP Source Setup Callbacks
        g_signal_connect(m_pSourceElement->GetGObject(), "pad-added", 
            G_CALLBACK(RtspSourceElementOnPadAddedCB), this);

        // Connect Decode Setup Callbacks
        g_signal_connect(m_pDecodeBin->GetGObject(), "pad-added", 
            G_CALLBACK(RtspDecodeElementOnPadAddedCB), this);
        g_signal_connect(m_pDecodeBin->GetGObject(), "child-added", 
            G_CALLBACK(OnChildAddedCB), this);

        AddChild(m_pPreDecodeTee);
        AddChild(m_pPreDecodeQueue);
        AddChild(m_pDecodeBin);
        AddChild(m_pSourceQueue);

        // Source Ghost Pad for Source Queue as src pad to connect to streammuxer
        m_pSourceQueue->AddGhostPadToParent("src");
        
        // New timestamp PPH to stamp the time of the last buffer - used to monitor the RTSP connection
        std::string handlerName = GetName() + "-timestamp-pph";
        m_TimestampPph = DSL_PPH_TIMESTAMP_NEW(handlerName.c_str());
        
        std::string padProbeName = GetName() + "-src-pad-probe";
        m_pSrcPadProbe = DSL_PAD_BUFFER_PROBE_NEW(padProbeName.c_str(), "src", m_pSourceQueue);
        m_pSrcPadProbe->AddPadProbeHandler(m_TimestampPph);
        
        g_mutex_init(&m_reconnectionMutex);
        
    }

    RtspSourceBintr::~RtspSourceBintr()
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            UnlinkAll();
        }
        if (m_streamMgtTimerId)
        {
            g_source_remove(m_streamMgtTimerId);
        }
        if (m_resetMgtTimerId)
        {
            g_source_remove(m_resetMgtTimerId);
        }

        m_pSrcPadProbe->RemovePadProbeHandler(m_TimestampPph);
        
        g_mutex_clear(&m_reconnectionMutex);
        
    }
    
    bool RtspSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("RtspSourceBintr '" << GetName() << "' is already in a linked state");
            return false;
        }

        if (HasTapBintr())
        {
            GstPadTemplate* pPadTemplate = 
                gst_element_class_get_pad_template(GST_ELEMENT_GET_CLASS(m_pPreDecodeTee->GetGstElement()), "src_%u");
            if (!pPadTemplate)
            {
                LOG_ERROR("Failed to get Pad Template for '" << GetName() << "'");
                return false;
            }
        
            // The TEE for this source is linked to both the "pre-decode-queue" and the TapBintr

            GstPad* pGstRequestedSourcePad = gst_element_request_pad(m_pPreDecodeTee->GetGstElement(), pPadTemplate, NULL, NULL);
            if (!pGstRequestedSourcePad)
            {
                LOG_ERROR("Failed to get Tee Pad for RTSP Source '" << GetName() <<"'");
                return false;
            }
            std::string padForPreDecodeName = "padForPreDecodeQueue_" + std::to_string(m_uniqueId);

            m_pGstRequestedSourcePads[padForPreDecodeName] = pGstRequestedSourcePad;
        
            if (!m_pTapBintr->LinkAll() or !m_pTapBintr->LinkToSource(m_pPreDecodeTee) or
                !m_pPreDecodeQueue->LinkToSource(m_pPreDecodeTee))
            {
                return false;
            }
        }
        if (!m_pPreDecodeQueue->LinkToSink(m_pDecodeBin))
        {
            return false;
        }

        // Note: we don't set the linked state until after 
        return true;
    }

    void RtspSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("RtspSourceBintr '" << GetName() << "' is not in a linked state");
            return;
        }
        
        if (m_streamMgtTimerId)
        {
            // Otherwise, we're disabling management with an interval of 0
            g_source_remove(m_streamMgtTimerId);
            m_streamMgtTimerId = 0;
            LOG_INFO("Stream management disabled for RTSP Source '" << GetName() << "'");
        }
        
        m_pPreDecodeQueue->UnlinkFromSink();
        if (HasTapBintr())
        {
            m_pTapBintr->UnlinkFromSource();
            m_pTapBintr->UnlinkAll();
            m_pPreDecodeQueue->UnlinkFromSource();
        }
        m_pParser->UnlinkFromSink();
        m_pDepay->UnlinkFromSink();
        
        for (auto const& imap: m_pGstRequestedSourcePads)
        {
            gst_element_release_request_pad(m_pTee->GetGstElement(), imap.second);
            gst_object_unref(imap.second);
        }
        
        m_isLinked = false;
    }

    bool RtspSourceBintr::SetUri(const char* uri)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Uri for RtspSourceBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }
        std::string newUri(uri);
        if (newUri.find("rtsp") == std::string::npos)
        {
            if (m_isLive)
            {
                LOG_ERROR("Invalid URI '" << uri << "' for Live source '" << GetName() << "'");
                return false;
            }
            std::ifstream streamUriFile(uri);
            if (!streamUriFile.good())
            {
                LOG_ERROR("URI Source'" << uri << "' Not found");
                return false;
            }
            // File source, not live - setup full path
            char absolutePath[PATH_MAX+1];
            m_uri.assign(realpath(uri, absolutePath));
            m_uri.insert(0, "file:");
        }        
        m_pSourceElement->SetAttribute("location", m_uri.c_str());
        
        return true;
    }
    
    uint RtspSourceBintr::GetBufferTimeout()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionMutex);
        
        return m_bufferTimeout;
    }
    
    void RtspSourceBintr::SetBufferTimeout(uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionMutex);
        
        if (m_bufferTimeout == timeout)
        {
            LOG_WARN("Buffer timeout for RTSP Source '" << GetName() << "' is already set to " << timeout);
            return;
        }
        m_bufferTimeout = timeout;

        // If we're all ready in a linked state, 
        if (IsLinked()) 
        {
            // If we're setting the reconect interval from zero to non-zero.
            if (timeout)
            {
                // Start up stream mangement
                m_streamMgtTimerId = g_timeout_add(1000, RtspStreamMgtHandler, this);
                LOG_INFO("Stream management enabled for RTSP Source '" << GetName() << "'");
            }
            else
            {
                // Otherwise, we're disabling management with an interval of 0
                g_source_remove(m_streamMgtTimerId);
                m_streamMgtTimerId = 0;
                LOG_INFO("Stream management disabled for RTSP Source '" << GetName() << "'");
            }
        }
    }
    
    void RtspSourceBintr::GetReconnectStats(uint* lastTime, uint* lastCount)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionMutex);

        *lastTime = m_lastReconnectTime.tv_sec;
        *lastCount = m_lastReconnectCount;
    }
    
    void RtspSourceBintr::ClearReconnectStats()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionMutex);

        m_lastReconnectTime = {0,0};
        m_lastReconnectCount = 0;
    }

    void  RtspSourceBintr::GetResetStats(boolean* isInReset, uint* resetCount)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionMutex);
        
        *isInReset = m_isInReset;
        *resetCount = m_lastResetCount;
    }

    bool RtspSourceBintr::AddTapBintr(DSL_BASE_PTR pTapBintr)
    {
        LOG_FUNC();
        
        if (m_pTapBintr)
        {
            LOG_ERROR("Source '" << GetName() << "' allready has a Tap");
            return false;
        }
        m_pTapBintr = std::dynamic_pointer_cast<TapBintr>(pTapBintr);
        AddChild(pTapBintr);
        return true;
    }

    bool RtspSourceBintr::RemoveTapBintr()
    {
        LOG_FUNC();

        if (!m_pTapBintr)
        {
            LOG_ERROR("Source '" << GetName() << "' does not have a Tap");
            return false;
        }
        RemoveChild(m_pTapBintr);
        m_pTapBintr = nullptr;
        return true;
    }
    
    bool RtspSourceBintr::HasTapBintr()
    {
        LOG_FUNC();
        
        return (m_pTapBintr != nullptr);
    }
    
    bool RtspSourceBintr::HandleSelectStream(GstElement *pBin, uint num, GstCaps *caps)
    {
        GstStructure *structure = gst_caps_get_structure(caps, 0);
        std::string media = gst_structure_get_string (structure, "media");
        std::string encoding = gst_structure_get_string (structure, "encoding-name");

        LOG_INFO("Media = '" << media << "' for RtspSourceBitnr '" << GetName() << "'");
        LOG_INFO("Encoding = '" << encoding << "' for RtspSourceBitnr '" << GetName() << "'");

        if (media.find("video") == std::string::npos)
        {
            LOG_WARN("Unsupported media = '" << media << "' for RtspSourceBitnr '" << GetName() << "'");
            return false;
        }
        if (encoding.find("H264") != std::string::npos)
        {
            m_pParser = DSL_ELEMENT_NEW("h264parse", "src-parse");
            m_pDepay = DSL_ELEMENT_NEW("rtph264depay", "src-depay");
        }
        else if (encoding.find("H265") != std::string::npos)
        {
            m_pParser = DSL_ELEMENT_NEW("h265parse", "src-parse");
            m_pDepay = DSL_ELEMENT_NEW("rtph265depay", "src-depayload");
        }
        else
        {
            LOG_ERROR("Unsupported encoding = '" << encoding << "' for RtspSourceBitnr '" << GetName() << "'");
            return false;
        }
        AddChild(m_pDepay);
        AddChild(m_pParser);

        // If we're tapping off of the pre-decode source stream, then link to the pre-decode Tee
        // The Pre-decode Queue will already be linked downstream as the first branch on the Tee
        if (HasTapBintr())
        {
            if (!m_pDepay->LinkToSink(m_pParser) or !m_pParser->LinkToSink(m_pPreDecodeTee))
            {
                return false;
            }            
        }
        // otherwise, there is no Tee and we link to the Pre-decode Queue directly
        else
        {
            if (!m_pDepay->LinkToSink(m_pParser) or !m_pParser->LinkToSink(m_pPreDecodeQueue))
            {
                return false;
            }            
        }
        if (!gst_element_sync_state_with_parent(m_pDepay->GetGstElement()) or
            !gst_element_sync_state_with_parent(m_pParser->GetGstElement()))
        {
            LOG_ERROR("Failed to sync Parser/Decoder states with Parent for RtspSourceBitnr '" << GetName() << "'");
            return false;
        }
        return true;
    }
        
    void RtspSourceBintr::HandleSourceElementOnPadAdded(GstElement* pBin, GstPad* pPad)
    {
        LOG_FUNC();

        GstCaps* pCaps = gst_pad_query_caps(pPad, NULL);
        GstStructure* structure = gst_caps_get_structure(pCaps, 0);
        std::string name = gst_structure_get_name(structure);
        
        LOG_INFO("Caps structs name " << name);
        if (name.find("x-rtp") != std::string::npos)
        {
            m_pGstStaticSinkPad = gst_element_get_static_pad(m_pDepay->GetGstElement(), "sink");
            if (!m_pGstStaticSinkPad)
            {
                LOG_ERROR("Failed to get Static Source Pad for Streaming Source '" << GetName() << "'");
            }
            
            if (gst_pad_link(pPad, m_pGstStaticSinkPad) != GST_PAD_LINK_OK) 
            {
                LOG_ERROR("Failed to link de-payload to pipeline");
                throw;
            }
            
            LOG_INFO("Video decode linked for URI source '" << GetName() << "'");
        }
    }
    
    void RtspSourceBintr::HandleDecodeElementOnPadAdded(GstElement* pBin, GstPad* pPad)
    {
        LOG_FUNC();

        GstCaps* pCaps = gst_pad_query_caps(pPad, NULL);
        GstStructure* structure = gst_caps_get_structure(pCaps, 0);
        std::string name = gst_structure_get_name(structure);
        
        LOG_INFO("Caps structs name " << name);
        if (name.find("video") != std::string::npos)
        {
            m_pGstStaticSinkPad = gst_element_get_static_pad(m_pSourceQueue->GetGstElement(), "sink");
            if (!m_pGstStaticSinkPad)
            {
                LOG_ERROR("Failed to get Static Source Pad for RTSP Source '" << GetName() << "'");
            }
            
            if (gst_pad_link(pPad, m_pGstStaticSinkPad) != GST_PAD_LINK_OK) 
            {
                LOG_ERROR("Failed to link decodebin to pipeline");
                throw;
            }
            
            // Update the cap memebers for this URI Source Bintr
            gst_structure_get_uint(structure, "width", &m_width);
            gst_structure_get_uint(structure, "height", &m_height);
            gst_structure_get_fraction(structure, "framerate", (gint*)&m_fps_n, (gint*)&m_fps_d);
            
            // Start the Stream mangement timer.
            if (m_bufferTimeout)
            {
                m_streamMgtTimerId = g_timeout_add(1000, RtspStreamMgtHandler, this);
                LOG_INFO("Starting stream management for RTSP Source '" << GetName() << "'");
            }

            // Now fully linked
            m_isLinked = true;

            LOG_INFO("Video decode linked for RTSP Source '" << GetName() << "'");
        }
    }

    int RtspSourceBintr::ManageStream()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionMutex);

        // if currently in a reset cycle then let the ResetStream handler continue to handle
        if (m_isInReset)
        {
            return true;
        }

        struct timeval currentTime;
        gettimeofday(&currentTime, NULL);
        
        // m_lastResetTime set to 0,0 on construction, and updated in Reset() 
        double timeSinceLastResetMs = (1000.0 * (currentTime.tv_sec - m_lastResetTime.tv_sec)) +
            ((currentTime.tv_usec - m_lastResetTime.tv_usec) / 1000.0);
            
        // Get the last buffer time. This timer callback should not be called until after the timer 
        // is started on successful linkup - therefore the lastBufferTime should be non-zero
        struct timeval lastBufferTime;
        m_TimestampPph->GetTime(lastBufferTime);
        if (lastBufferTime.tv_sec == 0)
        {
            LOG_ERROR("ManageStream callback called before the connection has been establed for source '" << GetName() << "'");
            return false;
        }

        double timeSinceLastBufferMs = 1000.0*(currentTime.tv_sec - lastBufferTime.tv_sec) + 
            (currentTime.tv_usec - lastBufferTime.tv_usec) / 1000.0;

        if (timeSinceLastBufferMs > m_bufferTimeout*1000)
        {
            LOG_INFO("Buffer timeout of " << m_bufferTimeout << "exceeded for source '" << GetName() << "'");
            m_resetMgtTimerId = g_timeout_add(100, RtspStreamResetHandler, this);
        }
        return true;
    }
    
    int RtspSourceBintr::ResetStream()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionMutex);

        gettimeofday(&m_lastResetTime, NULL);
        
        uint stateResult(0);
        GstState currentState;
        
        if (!m_isInReset)
        {
            // set the reset-state,
            m_isInReset = true;
            m_lastResetCount++; 

            LOG_DEBUG("Resetting RTSP Source '" << GetName() << "' with reset count = " << m_lastResetCount);
            
            // update the current buffer timestamp to the current reset time
            m_TimestampPph->SetTime(m_lastResetTime);

            if (!SetState(GST_STATE_NULL))
            {
                LOG_ERROR("Failed to set RTSP Source '" << GetName() << "' state to GST_STATE_NULL");
                return false;
            }
            stateResult = SyncStateWithParent();
        }
        else
        {
            stateResult = GetState(currentState);
        }
            
        switch (stateResult) 
        {
            case GST_STATE_CHANGE_SUCCESS:
                LOG_INFO("Reset completed for RTSP Source'" << GetName() << "'");
                m_isInReset = false;
                m_lastResetCount = 0; 
                m_lastReconnectTime = m_lastResetTime;
                m_lastResetCount++;
                m_resetMgtTimerId = 0;
                return false;
            case GST_STATE_CHANGE_FAILURE:
            case GST_STATE_CHANGE_NO_PREROLL:
                LOG_ERROR("FAILURE occured when trying to sync state for RTSP Source '" << GetName() << "'");
                return false;
            case GST_STATE_CHANGE_ASYNC:
                LOG_INFO("State change will complete asynchronously for RTSP Source '" << GetName() << "'");
                return true;
            default:
                break;
        }
       
        return false;
        
    }
    
    static void UriSourceElementOnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource)
    {
        static_cast<UriSourceBintr*>(pSource)->HandleSourceElementOnPadAdded(pBin, pPad);
    }
    
    static boolean RtspSourceSelectStreamCB(GstElement *pBin, uint num, GstCaps *caps,
        gpointer pSource)
    {
        static_cast<RtspSourceBintr*>(pSource)->HandleSelectStream(pBin, num, caps);
    }
        
    static void RtspSourceElementOnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource)
    {
        static_cast<RtspSourceBintr*>(pSource)->HandleSourceElementOnPadAdded(pBin, pPad);
    }
    
    static void RtspDecodeElementOnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource)
    {
        static_cast<RtspSourceBintr*>(pSource)->HandleDecodeElementOnPadAdded(pBin, pPad);
    }
    
    static void OnChildAddedCB(GstChildProxy* pChildProxy, GObject* pObject,
        gchar* name, gpointer pSource)
    {
        static_cast<DecodeSourceBintr*>(pSource)->HandleOnChildAdded(pChildProxy, pObject, name);
    }
    
    static void OnSourceSetupCB(GstElement* pObject, GstElement* arg0, 
        gpointer pSource)
    {
        static_cast<DecodeSourceBintr*>(pSource)->HandleOnSourceSetup(pObject, arg0);
    }
    
    static GstPadProbeReturn StreamBufferRestartProbCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pSource)
    {
        return static_cast<DecodeSourceBintr*>(pSource)->
            HandleStreamBufferRestart(pPad, pInfo);
    }

    static gboolean StreamBufferSeekCB(gpointer pSource)
    {
        return static_cast<DecodeSourceBintr*>(pSource)->HandleStreamBufferSeek();
    }

    static int RtspStreamMgtHandler(void* pSource)
    {
        return static_cast<RtspSourceBintr*>(pSource)->
            ManageStream();
    }

    static int RtspStreamResetHandler(void* pSource)
    {
        return static_cast<RtspSourceBintr*>(pSource)->
            ResetStream();
    }


} // SDL namespace