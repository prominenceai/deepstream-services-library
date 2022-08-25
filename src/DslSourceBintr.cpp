/*
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
#include "DslSourceBintr.h"
#include "DslPipelineBintr.h"
#include "DslSurfaceTransform.h"
#include <nvdsgstutils.h>

#define N_DECODE_SURFACES 16
#define N_EXTRA_SURFACES 1

namespace DSL
{
    SourceBintr::SourceBintr(const char* name)
        : Bintr(name)
        , m_cudaDeviceProp{0}
        , m_isLive(true)
        , m_width(0)
        , m_height(0)
        , m_fpsN(0)
        , m_fpsD(0)
        , m_latency(100)
        , m_numDecodeSurfaces(N_DECODE_SURFACES)
        , m_numExtraSurfaces(N_EXTRA_SURFACES)
    {
        LOG_FUNC();

        // Set the stream-id of the unique Source name
        SetId(Services::GetServices()->_sourceNameSet(name));

            // Get the Device properties
        cudaGetDeviceProperties(&m_cudaDeviceProp, m_gpuId);
    }
    
    SourceBintr::~SourceBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
        
        Services::GetServices()->_sourceNameErase(GetCStrName());
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
            LOG_ERROR("Source '" << GetName() << "' is not a child of Pipeline '" 
                << pParentBintr->GetName() << "'");
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

    void  SourceBintr::GetFrameRate(uint* fpsN, uint* fpsD)
    {
        LOG_FUNC();
        
        *fpsN = m_fpsN;
        *fpsD = m_fpsD;
    }

    //*********************************************************************************

    CsiSourceBintr::CsiSourceBintr(const char* name, 
        guint width, guint height, guint fpsN, guint fpsD)
        : SourceBintr(name)
        , m_sensorId(0)
    {
        LOG_FUNC();

        m_width = width;
        m_height = height;
        m_fpsN = fpsN;
        m_fpsD = fpsD;
        
        m_pSourceElement = DSL_ELEMENT_NEW("nvarguscamerasrc", name);
        m_pCapsFilter = DSL_ELEMENT_NEW("capsfilter", name);

        // aarch64
        if (m_cudaDeviceProp.integrated)
        {
            m_pSourceElement->SetAttribute("sensor-id", m_sensorId);
            m_pSourceElement->SetAttribute("bufapi-version", TRUE);
        }
        
        GstCaps * pCaps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12",
            "width", G_TYPE_INT, m_width, "height", G_TYPE_INT, m_height, 
            "framerate", GST_TYPE_FRACTION, m_fpsN, m_fpsD, NULL);
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
        guint width, guint height, guint fpsN, guint fpsD)
        : SourceBintr(name)
        , m_sensorId(0)
    {
        LOG_FUNC();

        m_width = width;
        m_height = height;
        m_fpsN = fpsN;
        m_fpsD = fpsD;
        
        m_pSourceElement = DSL_ELEMENT_NEW("v4l2src", name);
        m_pCapsFilter = DSL_ELEMENT_NEW("capsfilter", name);

        if (!m_cudaDeviceProp.integrated)
        {
            m_pVidConv1 = DSL_ELEMENT_EXT_NEW("nvvideoconvert", name, "1");
            AddChild(m_pVidConv1);
        }
        m_pVidConv2 = DSL_ELEMENT_EXT_NEW("nvvideoconvert", name, "2");

        GstCaps * pCaps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12",
            "width", G_TYPE_INT, m_width, "height", G_TYPE_INT, m_height, 
            "framerate", GST_TYPE_FRACTION, m_fpsN, m_fpsD, NULL);
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
        m_pVidConv2->SetAttribute("nvbuf-memory-type", m_nvbufMemType);

        AddChild(m_pSourceElement);
        AddChild(m_pCapsFilter);
        AddChild(m_pVidConv2);
        
        m_pCapsFilter->AddGhostPadToParent("src");
    }

    UsbSourceBintr::~UsbSourceBintr()
    {
        LOG_FUNC();
    }

    bool UsbSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("UsbSourceBintr '" << GetName() << "' is already in a linked state");
            return false;
        }
        
        // x86_64
        if (!m_cudaDeviceProp.integrated)
        {
            if (!m_pSourceElement->LinkToSink(m_pVidConv1) or 
                !m_pVidConv1->LinkToSink(m_pVidConv2) or
                !m_pVidConv2->LinkToSink(m_pCapsFilter))
            {
                return false;
            }
        }
        else // aarch_64
        {
            if (!m_pSourceElement->LinkToSink(m_pVidConv2) or 
                !m_pVidConv2->LinkToSink(m_pCapsFilter))
            {
                return false;
            }
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
        
        // x86_64
        if (!m_cudaDeviceProp.integrated)
        {
            m_pVidConv1->UnlinkFromSink();
        }
        m_pVidConv2->UnlinkFromSink();
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
        LOG_DEBUG("Setting GPU ID to '" << gpuId 
            << "' for UsbSourceBintr '" << m_name << "'");

        m_pVidConv2->SetAttribute("gpu-id", m_gpuId);
        
        return true;
    }

    //*********************************************************************************

    DecodeSourceBintr::DecodeSourceBintr(const char* name, 
        const char* factoryName, const char* uri,
        bool isLive, uint intraDecode, uint dropFrameInterval)
        : ResourceSourceBintr(name, uri)
        , m_cudadecMemtype(DSL_NVBUF_MEM_TYPE_DEFAULT)
        , m_intraDecode(intraDecode)
        , m_dropFrameInterval(dropFrameInterval)
        , m_accumulatedBase(0)
        , m_prevAccumulatedBase(0)
        , m_pDecoderStaticSinkpad(NULL)
        , m_bufferProbeId(0)
        , m_repeatEnabled(false)
    {
        LOG_FUNC();
        
        m_isLive = isLive;
        
        // Initialize the mutex regardless of IsLive or not
        g_mutex_init(&m_repeatEnabledMutex);

        m_pSourceElement = DSL_ELEMENT_NEW(factoryName, name);
        
        // if it's a file source, 
        if ((m_uri.find("http") == std::string::npos) and (m_uri.find("rtsp") == std::string::npos))
        {
            if (isLive)
            {
                LOG_ERROR("Invalid URI '" << uri << "' for Live source '" << name << "'");
                throw;
            }
            // Setup the absolute File URI and query dimensions
            if (!SetFileUri(uri))
            {
                LOG_ERROR("URI Source'" << uri << "' Not found");
                throw;
            }
        }
        
        LOG_INFO("URI Path for File Source '" << GetName() << "' = " << m_uri);
        
        if (m_uri.find("rtsp") != std::string::npos)
        {
            // Configure the source to generate NTP sync values
            configure_source_for_ntp_sync(m_pSourceElement->GetGstElement());
            m_pSourceElement->SetAttribute("location", m_uri.c_str());
        }
        else
        {
            // File Source may have empty URI (a.k.a file_path), in which case we
            // hold off setting the Source Element untill path is set.
            if (m_uri.size())
            {
                m_pSourceElement->SetAttribute("uri", m_uri.c_str());
            }
        }
        AddChild(m_pSourceElement);
    }
    
    DecodeSourceBintr::~DecodeSourceBintr()
    {
        LOG_FUNC();
 
        //DisableEosConsumer();
        g_mutex_clear(&m_repeatEnabledMutex);
    }
    
    bool DecodeSourceBintr::SetFileUri(const char* uri)
    {
        LOG_FUNC();

        std::string testUri(uri);
        if (testUri.empty())
        {
            LOG_INFO("File Path for SourceBintr '" << GetName() 
                << "' is empty. Source is in a non playable state");
            return true;
        }

        std::ifstream streamUriFile(uri);
        if (!streamUriFile.good())
        {
            LOG_ERROR("File Source '" << uri << "' Not found");
            return false;
        }
        // File source, not live - setup full path
        char absolutePath[PATH_MAX+1];
        m_uri.assign(realpath(uri, absolutePath));
        m_uri.insert(0, "file:");

        LOG_INFO("File Path = " << m_uri);
        
        // use openCV to open the file and read the Frame width and height properties.
        cv::VideoCapture vidCap;
        vidCap.open(uri, cv::CAP_ANY);

        if (!vidCap.isOpened())
        {
            LOG_ERROR("Failed to open File '" << uri 
                << "' for VideoRenderPlayerBintr '" << GetName() << "'");
            return false;
        }
        m_width = vidCap.get(cv::CAP_PROP_FRAME_WIDTH);
        m_height = vidCap.get(cv::CAP_PROP_FRAME_HEIGHT);
        
        // Note: the m_fpsN and m_fpsD can be calculated from cv.CAP_PROP_FPS
        // if needed prior to playing the file.
        return true;
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
            // aarch64 only
            if (m_cudaDeviceProp.integrated)
            {
                g_object_set(pObject, "enable-max-performance", TRUE, NULL);
            }
            g_object_set(pObject, "drop-frame-interval", m_dropFrameInterval, NULL);
            g_object_set(pObject, "num-extra-surfaces", m_numExtraSurfaces, NULL);

            // if the source is from file, then setup Stream buffer probe function
            // to handle the stream restart/loop on GST_EVENT_EOS.
            if (!m_isLive and m_repeatEnabled)
            {
                GstPadProbeType mask = (GstPadProbeType) 
                    (GST_PAD_PROBE_TYPE_EVENT_BOTH |
                    GST_PAD_PROBE_TYPE_EVENT_FLUSH | 
                    GST_PAD_PROBE_TYPE_BUFFER);
                    
                m_pDecoderStaticSinkpad = 
                    gst_element_get_static_pad(GST_ELEMENT(pObject), "sink");
                
                m_bufferProbeId = gst_pad_add_probe(m_pDecoderStaticSinkpad, 
                    mask, StreamBufferRestartProbCB, this, NULL);
            }
        }
    }
    
    GstPadProbeReturn DecodeSourceBintr::HandleStreamBufferRestart(GstPad* pPad, 
        GstPadProbeInfo* pInfo)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_repeatEnabledMutex);
        
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
        SetState(GST_STATE_PAUSED, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND);
        
        gboolean retval = gst_element_seek(GetGstElement(), 1.0, GST_FORMAT_TIME,
            (GstSeekFlags)(GST_SEEK_FLAG_KEY_UNIT | GST_SEEK_FLAG_FLUSH),
            GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_NONE, GST_CLOCK_TIME_NONE);

        if (!retval)
        {
            LOG_WARN("Failure to seek");
        }

        SetState(GST_STATE_PLAYING, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND);
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
    
    void DecodeSourceBintr::DisableEosConsumer()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_repeatEnabledMutex);
        
        if (m_pDecoderStaticSinkpad)
        {
            if (m_bufferProbeId)
            {
                gst_pad_remove_probe(m_pDecoderStaticSinkpad, m_bufferProbeId);
            }
            gst_object_unref(m_pDecoderStaticSinkpad);
        }
    }
    

    //*********************************************************************************

    UriSourceBintr::UriSourceBintr(const char* name, const char* uri, bool isLive,
        uint intraDecode, uint dropFrameInterval)
        : DecodeSourceBintr(name, "uridecodebin", uri, 
            isLive, intraDecode, dropFrameInterval)
    {
        LOG_FUNC();
        
        // New Elementrs for this Source
        m_pSourceQueue = DSL_ELEMENT_EXT_NEW("queue", name, "src");
        m_pTee = DSL_ELEMENT_NEW("tee", name);
        m_pFakeSinkQueue = DSL_ELEMENT_EXT_NEW("queue", name, "fakesink");
        m_pFakeSink = DSL_ELEMENT_NEW("fakesink", name);

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
            gst_element_class_get_pad_template(
                GST_ELEMENT_GET_CLASS(m_pTee->GetGstElement()), "src_%u");
        if (!pPadTemplate)
        {
            LOG_ERROR("Failed to get Pad Template for '" << GetName() << "'");
            return false;
        }
        
        // The TEE for this source is linked to both the "source queue" and "fake sink queue"

        GstPad* pGstRequestedSourcePad = gst_element_request_pad(m_pTee->GetGstElement(), 
            pPadTemplate, NULL, NULL);
        if (!pGstRequestedSourcePad)
        {
            LOG_ERROR("Failed to get Tee Pad for PipelineSinksBintr '" << GetName() <<"'");
            return false;
        }
        std::string padForSourceQueueName = "padForSourceQueue_" + std::to_string(m_uniqueId);

        m_pGstRequestedSourcePads[padForSourceQueueName] = pGstRequestedSourcePad;
        
        if (HasDewarperBintr())
        {
            if (!m_pDewarperBintr->LinkToSource(m_pTee) or 
                !m_pDewarperBintr->LinkToSink(m_pSourceQueue))
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

        pGstRequestedSourcePad = gst_element_request_pad(m_pTee->GetGstElement(), 
            pPadTemplate, NULL, NULL);
        if (!pGstRequestedSourcePad)
        {
            LOG_ERROR("Failed to get Tee Pad for PipelineSinksBintr '" << GetName() <<"'");
            return false;
        }
        std::string padForFakeSinkQueueName = "padForFakeSinkQueue_" + std::to_string(m_uniqueId);

        m_pGstRequestedSourcePads[padForFakeSinkQueueName] = pGstRequestedSourcePad;

        if (!m_pFakeSinkQueue->LinkToSource(m_pTee) or 
            !m_pFakeSinkQueue->LinkToSink(m_pFakeSink))
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
                LOG_ERROR("Failed to get Static Source Pad for Streaming Source '" 
                    << GetName() << "'");
            }
            
            if (gst_pad_link(pPad, m_pGstStaticSinkPad) != GST_PAD_LINK_OK) 
            {
                LOG_ERROR("Failed to link decodebin to source Tee");
                throw;
            }
            
            // Update the cap memebers for this URI Source Bintr
            gst_structure_get_uint(structure, "width", &m_width);
            gst_structure_get_uint(structure, "height", &m_height);
            gst_structure_get_fraction(structure, "framerate", (gint*)&m_fpsN, (gint*)&m_fpsD);
            
            LOG_INFO("Video decode linked for URI source '" << GetName() << "'");
        }
    }

    bool UriSourceBintr::SetUri(const char* uri)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Uri for UriSourceBintr '" << GetName() 
                << "' as it's currently Linked");
            return false;
        }
        // if it's a file source, 
        std::string newUri(uri);
        
        if ((newUri.find("http") == std::string::npos))
        {
            // Setup the absolute File URI and query dimensions
            if (!SetFileUri(uri))
            {
                LOG_ERROR("URI Source'" << uri << "' Not found");
                return false;
            }
        }        
        LOG_INFO("URI Path for File Source '" << GetName() << "' = " << m_uri);
        
        if (m_uri.size())
        {
            m_pSourceElement->SetAttribute("uri", m_uri.c_str());
        }
        
        return true;
    }

    //*********************************************************************************

    FileSourceBintr::FileSourceBintr(const char* name, 
        const char* uri, bool repeatEnabled)
        : UriSourceBintr(name, uri, false, false, 0)
    {
        LOG_FUNC();
        
        // override the default
        m_repeatEnabled = repeatEnabled;
    }
    
    FileSourceBintr::~FileSourceBintr()
    {
        LOG_FUNC();
    }

    bool FileSourceBintr::SetUri(const char* uri)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set File Path for FileSourceBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        
        if (!SetFileUri(uri))
        {
            return false;
        }
        if (m_uri.size())
        {
            m_pSourceElement->SetAttribute("uri", m_uri.c_str());
        }
        return true;
    }
    
    bool FileSourceBintr::GetRepeatEnabled()
    {
        LOG_FUNC();
        
        return m_repeatEnabled;
    }

    bool FileSourceBintr::SetRepeatEnabled(bool enabled)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Cannot set Repeat Enabled for Source '" << GetName() 
                << "' as it is currently Linked");
            return false;
        }
        
        m_repeatEnabled = enabled;
        return true;
    }

    //*********************************************************************************

    ImageSourceBintr::ImageSourceBintr(const char* name, const char* uri, uint type)
        : ResourceSourceBintr(name, uri)
    {
        LOG_FUNC();
        
        // override the default source attributes
        m_isLive = False;

        // Other components are created conditionaly by file type. 
        if (m_uri.find("jpeg") != std::string::npos or
            m_uri.find("jpg") != std::string::npos)
        {
            LOG_INFO("Setting file format to JPG for ImageSourceBintr '" 
                << GetName() << "'");
            m_format = DSL_IMAGE_FORMAT_JPG;
            m_ext = DSL_IMAGE_EXT_JPG;
            m_pParser = DSL_ELEMENT_NEW("jpegparse", name);
            m_pDecoder = DSL_ELEMENT_NEW("nvv4l2decoder", name); 
            
            AddChild(m_pDecoder);
            AddChild(m_pParser);
            
            // Source Ghost Pad for JPEG image sources
            m_pDecoder->AddGhostPadToParent("src");
            
            // If it's an MJPG file or Multi JPG files
            if (m_uri.find("mjpeg") != std::string::npos or
                m_uri.find("mjpg") != std::string::npos or
                type == DSL_IMAGE_TYPE_MULTI)
            {
                LOG_INFO("Setting decoder 'mjpeg' attribute for ImageSourceBintr '" 
                    << GetName() << "'");
                m_pDecoder->SetAttribute("mjpeg", true);
            }
            
        }
        else if (m_uri.find(".png") != std::string::npos)
        {
            LOG_ERROR("Unsuported file type (.png ) '" << m_uri 
                << "' for new Image Source '" << name << "'");
            throw;
        }
        else
        {
            LOG_ERROR("Invalid file type = '" << m_uri 
                << "' for new Image Source '" << name << "'");
            throw;
        }
    }
    
    ImageSourceBintr::~ImageSourceBintr()
    {
        LOG_FUNC();
    }

    bool ImageSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("ImageSourceBintr '" << GetName() << "' is already in a linked state");
            return false;
        }
        if (!IsLinkable())
        {
            LOG_ERROR("Unable to Link ImageSourceBintr '" << GetName() 
                << "' as its uri has not been set");
            return false;
        }
//        if (m_format == DSL_IMAGE_FORMAT_JPG)
//        {
            if (!m_pSourceElement->LinkToSink(m_pParser) or
                !m_pParser->LinkToSink(m_pDecoder))
            {
                LOG_ERROR("ImageSourceBintr '" << GetName() << "' failed to LinkAll");
                return false;
            }
//        }
//        else
//        {
//            // TODO
//        }
        m_isLinked = true;
        
        return true;
    }

    void ImageSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("ImageSourceBintr '" << GetName() 
                << "' is not in a linked state");
            return;
        }
        
        if (m_format == DSL_IMAGE_FORMAT_JPG)
        {
            if (!m_pSourceElement->UnlinkFromSink() or
                !m_pParser->UnlinkFromSink())
            {
                LOG_ERROR("ImageSourceBintr '" << GetName() 
                    << "' failed to UnlinkAll");
                return;
            }    
        }
        else
        {
            // TODO
        }
        m_isLinked = false;
    }

    //*********************************************************************************

    SingleImageSourceBintr::SingleImageSourceBintr(const char* name, const char* uri)
        : ImageSourceBintr(name, uri, DSL_IMAGE_TYPE_SINGLE)
    {
        LOG_FUNC();
        
        m_pSourceElement = DSL_ELEMENT_NEW("filesrc", name);
        AddChild(m_pSourceElement);
        
        if (!SetUri(uri))
        {
            throw;
        }

    }
    
    SingleImageSourceBintr::~SingleImageSourceBintr()
    {
        LOG_FUNC();
    }

    bool SingleImageSourceBintr::SetUri(const char* uri)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set File Path for ImageFrameSourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        
        std::string pathString(uri);
        if (pathString.empty())
        {
            LOG_INFO("File Path for ImageFrameSourceBintr '" << GetName() 
                << "' is empty. Source is in a non playable state");
            return true;
        }
        
        std::ifstream streamUriFile(uri);
        if (!streamUriFile.good())
        {
            LOG_ERROR("Image Source'" << uri << "' Not found");
            return false;
        }
        // File source, not live - setup full path
        char absolutePath[PATH_MAX+1];
        m_uri.assign(realpath(uri, absolutePath));

        // Use OpenCV to determine the new image dimensions
        cv::Mat image = imread(m_uri, cv::IMREAD_COLOR);
        cv::Size imageSize = image.size();
        m_width = imageSize.width;
        m_height = imageSize.height;

        // Set the filepath for the File Source Elementr
        m_pSourceElement->SetAttribute("location", m_uri.c_str());

        return true;
            
    }

    //*********************************************************************************

    MultiImageSourceBintr::MultiImageSourceBintr(const char* name, 
        const char* uri, uint fpsN, uint fpsD)
        : ImageSourceBintr(name, uri, DSL_IMAGE_TYPE_MULTI)
    {
        LOG_FUNC();
        
        // override the default source attributes
        m_fpsN = fpsN;
        m_fpsD = fpsD;

        m_pSourceElement = DSL_ELEMENT_NEW("multifilesrc", name);
        AddChild(m_pSourceElement);

        if (!SetUri(uri))
        {
            throw;
        }

//        GstCaps * pCaps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12",
//            "width", G_TYPE_INT, m_width, "height", G_TYPE_INT, m_height, 
//            "framerate", GST_TYPE_FRACTION, m_fpsN, m_fpsD, NULL);
//        GstCaps * pCaps = gst_caps_new_simple("image/jpeg", "framerate", 
//            GST_TYPE_FRACTION, m_fpsN, m_fpsD, NULL);
//        if (!pCaps)
//        {
//            LOG_ERROR("Failed to create new Simple Capabilities for '" << name << "'");
//            throw;  
//        }

//        GstCapsFeatures *feature = NULL;
//        feature = gst_caps_features_new("memory:NVMM", NULL);
//        gst_caps_set_features(pCaps, 0, feature);

//        m_pSourceElement->SetAttribute("caps", pCaps);
//        
//        gst_caps_unref(pCaps);        
    }
    
    MultiImageSourceBintr::~MultiImageSourceBintr()
    {
        LOG_FUNC();
    }

    bool MultiImageSourceBintr::SetUri(const char* uri)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set File Path for ImageFrameSourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        
        std::string pathString(uri);
        if (pathString.empty())
        {
            LOG_INFO("File Path for ImageFrameSourceBintr '" << GetName() 
                << "' is empty. Source is in a non playable state");
            return true;
        }
        
//        if (m_type == DSL_IMAGE_TYPE_SINGLE)
//        {
//            std::ifstream streamUriFile(uri);
//            if (!streamUriFile.good())
//            {
//                LOG_ERROR("Image Source'" << uri << "' Not found");
//                return false;
//            }
//            // File source, not live - setup full path
//            char absolutePath[PATH_MAX+1];
//            m_uri.assign(realpath(uri, absolutePath));
//
//            // Use OpenCV to determine the new image dimensions
//            cv::Mat image = imread(m_uri, cv::IMREAD_COLOR);
//            cv::Size imageSize = image.size();
//            m_width = imageSize.width;
//            m_height = imageSize.height;
//        }
        m_uri.assign(uri);
        // Set the filepath for the File Source Elementr
        m_pSourceElement->SetAttribute("location", m_uri.c_str());

        return true;
            
    }

    //*********************************************************************************

    ImageStreamSourceBintr::ImageStreamSourceBintr(const char* name, 
        const char* uri, bool isLive, uint fpsN, uint fpsD, uint timeout)
        : ResourceSourceBintr(name, uri)
        , m_timeout(timeout)
        , m_timeoutTimerId(0)
    {
        LOG_FUNC();
        
        // override default values
        m_isLive = isLive;
        m_fpsN = fpsN;
        m_fpsD = fpsD;

        m_pSourceElement = DSL_ELEMENT_NEW("videotestsrc", name);
        m_pSourceCapsFilter = DSL_ELEMENT_EXT_NEW("capsfilter", name, "source");
        m_pImageOverlay = DSL_ELEMENT_NEW("gdkpixbufoverlay", name); 
        m_pVidConv = DSL_ELEMENT_NEW("nvvideoconvert", name);
        m_pCapsFilter = DSL_ELEMENT_EXT_NEW("capsfilter", name, "sink");

        m_pSourceElement->SetAttribute("pattern", 2); // 2 = black


//        GstCaps * pCaps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12",
//            "width", G_TYPE_INT, m_width, "height", G_TYPE_INT, m_height, 
//            "framerate", GST_TYPE_FRACTION, m_fpsN, m_fpsD, NULL);
        GstCaps * pCaps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12",
            "framerate", GST_TYPE_FRACTION, m_fpsN, m_fpsD, NULL);
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
        
        m_pVidConv->SetAttribute("gpu-id", m_gpuId);
        m_pVidConv->SetAttribute("nvbuf-memory-type", m_nvbufMemType);

        // Add all new Elementrs as Children to the SourceBintr
        AddChild(m_pSourceElement);
        AddChild(m_pSourceCapsFilter);
        AddChild(m_pImageOverlay);
        AddChild(m_pVidConv);
        AddChild(m_pCapsFilter);
        
        // Source Ghost Pad for ImageStreamSourceBintr
        m_pCapsFilter->AddGhostPadToParent("src");

        g_mutex_init(&m_timeoutTimerMutex);

        if(uri and !SetUri(uri))
        {
            throw;
        }
    }
    
    ImageStreamSourceBintr::~ImageStreamSourceBintr()
    {
        LOG_FUNC();
        
        g_mutex_clear(&m_timeoutTimerMutex);
    }

    bool ImageStreamSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("ImageStreamSourceBintr '" << GetName() << "' is already in a linked state");
            return false;
        }
        if (!m_pSourceElement->LinkToSink(m_pSourceCapsFilter) or
            !m_pSourceCapsFilter->LinkToSink(m_pImageOverlay) or
            !m_pImageOverlay->LinkToSink(m_pVidConv) or
            !m_pVidConv->LinkToSink(m_pCapsFilter))
        {
            LOG_ERROR("ImageStreamSourceBintr '" << GetName() << "' failed to LinkAll");
            return false;
        }
        m_isLinked = true;
        
        if (m_timeout)
        {
            m_timeoutTimerId = g_timeout_add(m_timeout*1000, 
                ImageSourceDisplayTimeoutHandler, this);
        }
        
        return true;
    }

    void ImageStreamSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("ImageStreamSourceBintr '" << GetName() << "' is not in a linked state");
            return;
        }
        if (m_timeoutTimerId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_timeoutTimerMutex);
            g_source_remove(m_timeoutTimerId);
            m_timeoutTimerId = 0;
        }
        
        if (!m_pSourceElement->UnlinkFromSink() or
            !m_pSourceCapsFilter->UnlinkFromSink() or
            !m_pImageOverlay->UnlinkFromSink() or
            !m_pVidConv->UnlinkFromSink())
        {
            LOG_ERROR("ImageStreamSourceBintr '" << GetName() << "' failed to UnlinkAll");
            return;
        }    
        m_isLinked = false;
    }
    
    int ImageStreamSourceBintr::HandleDisplayTimeout()
    {
        LOG_FUNC();
        
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_timeoutTimerMutex);

        // Send the EOS event to end the Image display
        SendEos();
        m_timeoutTimerId = 0;
        
        // Single shot - so don't restart
        return 0;
    }

    bool ImageStreamSourceBintr::SetUri(const char* uri)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set File Path for ImageStreamSourceBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }
        std::string pathString(uri);
        if (pathString.empty())
        {
            LOG_INFO("File Path for ImageStreamSourceBintr '" << GetName() 
                << "' is empty. Source is in a non playable state");
            return true;
        }
            
        std::ifstream streamUriFile(uri);
        if (!streamUriFile.good())
        {
            LOG_ERROR("Image Source'" << uri << "' Not found");
            return false;
        }
        // File source, not live - setup full path
        char absolutePath[PATH_MAX+1];
        m_uri.assign(realpath(uri, absolutePath));

        // Use OpenCV to determine the new image dimensions
        cv::Mat image = imread(m_uri, cv::IMREAD_COLOR);
        cv::Size imageSize = image.size();
        m_width = imageSize.width;
        m_height = imageSize.height;

        GstCaps * pCaps = gst_caps_new_simple("video/x-raw", 
            "format", G_TYPE_STRING, "NV12", 
            "width", G_TYPE_INT, m_width, "height", G_TYPE_INT, m_height,
            "framerate", GST_TYPE_FRACTION, m_fpsN, m_fpsD, NULL);
        if (!pCaps)
        {
            LOG_ERROR("Failed to create new Simple Caps Filter for '" << m_name << "'");
            return false;  
        }
        m_pSourceCapsFilter->SetAttribute("caps", pCaps);
        gst_caps_unref(pCaps);        

        // Set the filepath for the Image Elementr
        m_pImageOverlay->SetAttribute("location", m_uri.c_str());
        
        return true;
    }
    
    uint ImageStreamSourceBintr::GetTimeout()
    {
        LOG_FUNC();
        
        return m_timeout;
    }

    bool ImageStreamSourceBintr::SetTimeout(uint timeout)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Cannot set Timeout for Image Source '" << GetName() 
                << "' as it is currently Linked");
            return false;
        }
        
        m_timeout = timeout;
        return true;
    }
    
    //*********************************************************************************

    InterpipeSourceBintr::InterpipeSourceBintr(const char* name, 
        const char* listenTo, bool isLive, bool acceptEos, bool acceptEvents)
        : SourceBintr(name)
        , m_listenTo(listenTo)
        , m_acceptEos(acceptEos)
        , m_acceptEvents(acceptEvents)
    {
        LOG_FUNC();
        
        // we need to append the factory name to match the Inter-Pipe
        // sinks element name. 
        m_listenToFullName = m_listenTo + "-interpipesink";
        
        LOG_INFO("listen-to sink name = " << m_listenToFullName);
        
        // override the default settings.
        m_isLive = isLive;
        
        m_pSourceElement = DSL_ELEMENT_NEW("interpipesrc", name);
        
        m_pSourceElement->SetAttribute("listen-to", m_listenToFullName.c_str());
        m_pSourceElement->SetAttribute("accept-eos-event", m_acceptEos);
        m_pSourceElement->SetAttribute("accept-events", m_acceptEvents);
        m_pSourceElement->SetAttribute("allow-renegotiation", true);

        // Add the new Elementr as a Child to the SourceBintr
        AddChild(m_pSourceElement);
        
        m_pSourceElement->AddGhostPadToParent("src");
    }
    
    InterpipeSourceBintr::~InterpipeSourceBintr()
    {
        LOG_FUNC();
    }

    const char* InterpipeSourceBintr::GetListenTo()
    {
        LOG_FUNC();
        
        return m_listenTo.c_str();
    }
    
    void InterpipeSourceBintr::SetListenTo(const char* listenTo)
    {
        m_listenTo = listenTo;
        m_listenToFullName = m_listenTo + "-interpipesink";
        
        m_pSourceElement->SetAttribute("listen-to", m_listenToFullName.c_str());
    }
    
    bool InterpipeSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("InterpipeSourceBintr '" << GetName() 
                << "' is already in a linked state");
            return false;
        }
        // Single element nothing to link
        m_isLinked = true;
        return true;
    }

    void InterpipeSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("InterpipeSourceBintr '" << GetName() 
                << "' is not in a linked state");
            return;
        }
        // Single element nothing to link
        m_isLinked = false;
    }
    
    void InterpipeSourceBintr::GetAcceptSettings(bool* acceptEos, 
        bool* acceptEvents)
    {
        LOG_FUNC();
        
        *acceptEos = m_acceptEos;
        *acceptEvents = m_acceptEvents;
    }

    bool InterpipeSourceBintr::SetAcceptSettings(bool acceptEos, 
        bool acceptEvents)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Accept setting for InterpipeSourceBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_acceptEos = acceptEos;
        m_acceptEvents = acceptEvents;
        
        m_pSourceElement->SetAttribute("accept-eos-event", m_acceptEos);
        m_pSourceElement->SetAttribute("accept-events", m_acceptEvents);
        
        return true;
    }
    
    //*********************************************************************************
    
    RtspSourceBintr::RtspSourceBintr(const char* name, const char* uri, 
        uint protocol, uint intraDecode, uint dropFrameInterval, 
        uint latency, uint timeout)
        : DecodeSourceBintr(name, "rtspsrc", uri, true, intraDecode, dropFrameInterval)
        , m_rtpProtocols(protocol)
        , m_bufferTimeout(timeout)
        , m_streamManagerTimerId(0)
        , m_reconnectionManagerTimerId(0)
        , m_connectionData{0}
        , m_reconnectionFailed(false)
        , m_reconnectionSleep(0)
        , m_reconnectionStartTime{0}
        , m_currentState(GST_STATE_NULL)
        , m_previousState(GST_STATE_NULL)
        , m_listenerNotifierTimerId(0)
    {
        LOG_FUNC();

        // Set RTSP latency
        m_latency = latency;

        // New RTSP Specific Elementrs for this Source
        m_pPreDecodeTee = DSL_ELEMENT_NEW("tee", name);
        m_pPreDecodeQueue = DSL_ELEMENT_EXT_NEW("queue", name, "decodebin");
        m_pDecodeBin = DSL_ELEMENT_NEW("decodebin", name);
        m_pSourceQueue = DSL_ELEMENT_EXT_NEW("queue", name, "src");

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
        
        // New timestamp PPH to stamp the time of the last buffer 
        // - used to monitor the RTSP connection
        std::string handlerName = GetName() + "-timestamp-pph";
        m_TimestampPph = DSL_PPH_TIMESTAMP_NEW(handlerName.c_str());
        
        std::string padProbeName = GetName() + "-src-pad-probe";
        m_pSrcPadProbe = DSL_PAD_BUFFER_PROBE_NEW(padProbeName.c_str(), "src", m_pSourceQueue);
        m_pSrcPadProbe->AddPadProbeHandler(m_TimestampPph);
        
        g_mutex_init(&m_streamManagerMutex);
        g_mutex_init(&m_reconnectionManagerMutex);
        g_mutex_init(&m_stateChangeMutex);
        
        // Set the default connection param values
        m_connectionData.sleep = DSL_RTSP_RECONNECTION_SLEEP_S;
        m_connectionData.timeout = DSL_RTSP_RECONNECTION_TIMEOUT_S;
    }

    RtspSourceBintr::~RtspSourceBintr()
    {
        LOG_FUNC();
        
        if (m_reconnectionManagerTimerId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionManagerMutex);
            g_source_remove(m_reconnectionManagerTimerId);
        }

        // Note: don't need t worry about stopping the one-shot m_listenerNotifierTimerId
        
        m_pSrcPadProbe->RemovePadProbeHandler(m_TimestampPph);
        
        g_mutex_clear(&m_streamManagerMutex);
        g_mutex_clear(&m_reconnectionManagerMutex);
        g_mutex_clear(&m_stateChangeMutex);
    }
    
    bool RtspSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("RtspSourceBintr '" << GetName() << "' is already in a linked state");
            return false;
        }

        // Note: this is a workaround for an NVIDIA bug. We need to test the stream before
        // we try and link any pads. Otherwise, unlinking a failed stream connection from 
        // the Streammuxer will result in a deadlock. Try to open the URL with open CV first.
        cv::VideoCapture capture(m_uri.c_str());

        if (!capture.isOpened())
        {
            LOG_ERROR("RtspSourceBintr '" << GetName() << "' failed to open stream for URI = "
                << m_uri.c_str());
            return false;
        }

        if (HasTapBintr())
        {
            if (!m_pTapBintr->LinkAll() or !m_pTapBintr->LinkToSourceTee(m_pPreDecodeTee) or
                !m_pPreDecodeQueue->LinkToSourceTee(m_pPreDecodeTee, "src_%u"))
            {
                return false;
            }
        }
        if (!m_pPreDecodeQueue->LinkToSink(m_pDecodeBin))
        {
            return false;
        }
        m_isLinked = true;
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
        
        if (m_streamManagerTimerId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);
            
            g_source_remove(m_streamManagerTimerId);
            m_streamManagerTimerId = 0;
            LOG_INFO("Stream management disabled for RTSP Source '" << GetName() << "'");
        }
        if (m_reconnectionManagerTimerId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionManagerMutex);

            g_source_remove(m_reconnectionManagerTimerId);
            m_reconnectionManagerTimerId = 0;
            LOG_INFO("Reconnection management disabled for RTSP Source '" << GetName() << "'");
        }
        
        m_pPreDecodeQueue->UnlinkFromSink();
        if (HasTapBintr())
        {
            m_pPreDecodeQueue->UnlinkFromSourceTee();
            m_pTapBintr->UnlinkAll();
            m_pTapBintr->UnlinkFromSourceTee();
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
            LOG_ERROR("Invalid URI '" << uri << "' for RTSP Source '" << GetName() << "'");
            return false;
        }        
        m_pSourceElement->SetAttribute("location", m_uri.c_str());
        
        return true;
    }
    
    uint RtspSourceBintr::GetBufferTimeout()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);
        
        return m_bufferTimeout;
    }
    
    void RtspSourceBintr::SetBufferTimeout(uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);
        
        if (m_bufferTimeout == timeout)
        {
            LOG_WARN("Buffer timeout for RTSP Source '" << GetName() 
                << "' is already set to " << timeout);
            return;
        }

        // If we're all ready in a linked state, 
        if (IsLinked()) 
        {
            // If stream management is currently running, shut it down regardless
            if (m_streamManagerTimerId)
            {
                // shutdown the current session
                g_source_remove(m_streamManagerTimerId);
                m_streamManagerTimerId = 0;
                LOG_INFO("Stream management disabled for RTSP Source '" << GetName() << "'");
            }
            // If we have a new timeout value, we can renable
            if (timeout)
            {
                // Start up stream mangement
                m_streamManagerTimerId = g_timeout_add(timeout, 
                    RtspReconnectionMangerHandler, this);
                LOG_INFO("Stream management enabled for RTSP Source '" 
                    << GetName() << "' with timeout = " << timeout);
            }
            // Else, the client is disabling stream mangagement. Shut down the 
            // reconnection cycle if running. 
            else if (m_reconnectionManagerTimerId)
            {
                LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionManagerMutex);
                // shutdown the current reconnection cycle
                g_source_remove(m_reconnectionManagerTimerId);
                m_reconnectionManagerTimerId = 0;
                LOG_INFO("Reconnection management disabled for RTSP Source '" << GetName() << "'");
            }
        }
        m_bufferTimeout = timeout;
    }

    void RtspSourceBintr::GetReconnectionParams(uint* sleep, uint* timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionManagerMutex);
        
        *sleep = m_connectionData.sleep;
        *timeout = m_connectionData.timeout;
    }
    
    bool RtspSourceBintr::SetReconnectionParams(uint sleep, uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionManagerMutex);
        
        if (!sleep or !timeout)
        {
            LOG_INFO("Invalid reconnection params for RTSP Source '" << GetName() << "'");
            return false;
        }

        m_connectionData.sleep = sleep;
        m_connectionData.timeout = timeout;
        return true;
    }

    void RtspSourceBintr::GetConnectionData(dsl_rtsp_connection_data* data)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);

        *data = m_connectionData;
    }
    
    void RtspSourceBintr::_setConnectionData(dsl_rtsp_connection_data data)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);
        
        m_connectionData = data;
    }
    
    void RtspSourceBintr::ClearConnectionStats()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);

        m_connectionData.first_connected = 0;
        m_connectionData.last_connected = 0;
        m_connectionData.last_disconnected = 0;
        m_connectionData.count = 0;
        m_connectionData.retries = 0;
    }

    bool RtspSourceBintr::AddStateChangeListener(dsl_state_change_listener_cb listener, void* userdata)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);
        
        if (m_stateChangeListeners.find(listener) != m_stateChangeListeners.end())
        {   
            LOG_ERROR("RTSP Source state-change-listener is not unique");
            return false;
        }
        m_stateChangeListeners[listener] = userdata;
        
        return true;
    }

    bool RtspSourceBintr::RemoveStateChangeListener(dsl_state_change_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);
        
        if (m_stateChangeListeners.find(listener) == m_stateChangeListeners.end())
        {   
            LOG_ERROR("RTSP Source state-change-listener");
            return false;
        }
        m_stateChangeListeners.erase(listener);
        
        return true;
    }
    
    bool RtspSourceBintr::AddTapBintr(DSL_BASE_PTR pTapBintr)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can not add Tap to Source '" << GetName() 
                << "' as it's in a Linked state");
            return false;
        }
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
        if (m_isLinked)
        {
            LOG_ERROR("Can not remove Tap from Source '" << GetName() 
                << "' as it's in a Linked state");
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

        if (!m_pParser)
        {
            if (media.find("video") == std::string::npos)
            {
                LOG_WARN("Unsupported media = '" << media << "' for RtspSourceBitnr '" 
                    << GetName() << "'");
                return false;
            }
            if (encoding.find("H264") != std::string::npos)
            {
                m_pParser = DSL_ELEMENT_NEW("h264parse", GetCStrName());
                m_pDepay = DSL_ELEMENT_NEW("rtph264depay", GetCStrName());
            }
            else if (encoding.find("H265") != std::string::npos)
            {
                m_pParser = DSL_ELEMENT_NEW("h265parse", GetCStrName());
                m_pDepay = DSL_ELEMENT_NEW("rtph265depay", GetCStrName());
            }
            else
            {
                LOG_ERROR("Unsupported encoding = '" << encoding << "' for RtspSourceBitnr '" 
                    << GetName() << "'");
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
                LOG_ERROR("Failed to sync Parser/Decoder states with Parent for RtspSourceBitnr '" 
                    << GetName() << "'");
                return false;
            }
        }
        return true;
    }
        
    void RtspSourceBintr::HandleSourceElementOnPadAdded(GstElement* pBin, GstPad* pPad)
    {
        LOG_FUNC();

        GstCaps* pCaps = gst_pad_query_caps(pPad, NULL);
        GstStructure* structure = gst_caps_get_structure(pCaps, 0);
        std::string name = gst_structure_get_name(structure);
        std::string media = gst_structure_get_string (structure, "media");
        std::string encoding = gst_structure_get_string (structure, "encoding-name");

        LOG_INFO("Caps structs name " << name);
        LOG_INFO("Media = '" << media << "' for RtspSourceBitnr '" << GetName() << "'");
        
        if (name.find("x-rtp") != std::string::npos and 
            media.find("video")!= std::string::npos)
        {
            m_pGstStaticSinkPad = gst_element_get_static_pad(m_pDepay->GetGstElement(), "sink");
            if (!m_pGstStaticSinkPad)
            {
                LOG_ERROR("Failed to get Static Source Pad for Streaming Source '" 
                    << GetName() << "'");
            }
            
            if (gst_pad_link(pPad, m_pGstStaticSinkPad) != GST_PAD_LINK_OK) 
            {
                LOG_ERROR("Failed to link source to de-payload");
                throw;
            }
            
            LOG_INFO("Video decode linked for RTSP source '" << GetName() << "'");
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
            m_pGstStaticSinkPad = 
                gst_element_get_static_pad(m_pSourceQueue->GetGstElement(), "sink");
            if (!m_pGstStaticSinkPad)
            {
                LOG_ERROR("Failed to get Static Source Pad for RTSP Source '" 
                    << GetName() << "'");
            }
            
            if (gst_pad_link(pPad, m_pGstStaticSinkPad) != GST_PAD_LINK_OK) 
            {
                LOG_ERROR("Failed to link decodebin to pipeline");
                throw;
            }
            
            // Update the cap memebers for this URI Source Bintr
            gst_structure_get_uint(structure, "width", &m_width);
            gst_structure_get_uint(structure, "height", &m_height);
            gst_structure_get_fraction(structure, "framerate", (gint*)&m_fpsN, (gint*)&m_fpsD);
            
            // Start the Stream mangement timer, only if timeout is enable and not currently running
            if (m_bufferTimeout and !m_streamManagerTimerId)
            {
                m_streamManagerTimerId = g_timeout_add(m_bufferTimeout, 
                    RtspStreamManagerHandler, this);
                LOG_INFO("Starting stream management for RTSP Source '" << GetName() << "'");
            }

            SetCurrentState(GST_STATE_READY);

            LOG_INFO("Video decode linked for RTSP Source '" << GetName() << "'");
        }
    }

    int RtspSourceBintr::StreamManager()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_streamManagerMutex);

        // if currently in a reset cycle then let the ResetStream handler continue to handle
        if (m_connectionData.is_in_reconnect)
        {
            return true;
        }

        struct timeval currentTime;
        gettimeofday(&currentTime, NULL);

        GstState currentState;
        uint stateResult = GetState(currentState, 0);
        SetCurrentState(currentState);
        
        // Get the last buffer time. This timer callback should not be called until after the timer 
        // is started on successful linkup - therefore the lastBufferTime should be non-zero
        struct timeval lastBufferTime;
        m_TimestampPph->GetTime(lastBufferTime);
        if (lastBufferTime.tv_sec == 0)
        {
            LOG_DEBUG("Waiting for first buffer before checking for timeout for source '" 
                << GetName() << "'");
            return true;
        }

        double timeSinceLastBufferMs = 1000.0*(currentTime.tv_sec - lastBufferTime.tv_sec) + 
            (currentTime.tv_usec - lastBufferTime.tv_usec) / 1000.0;

        if (timeSinceLastBufferMs < m_bufferTimeout*1000)
        {
            // Timeout has not been exceeded, so return true to sleep again
            return true;
        }
        LOG_INFO("Buffer timeout of " << m_bufferTimeout << " seconds exceeded for source '" 
            << GetName() << "'");
            
        if (HasTapBintr())
        {
            m_pTapBintr->HandleEos();
        }
        
        // Call the Reconnection Managter directly to start the reconnection cycle,
        if (!ReconnectionManager())
        {
            LOG_INFO("Unable to start re-connection manager for '" << GetName() << "'");
            return false;
        }
            
        LOG_INFO("Starting Re-connection Manager for source '" << GetName() << "'");
        m_reconnectionManagerTimerId = g_timeout_add(1000, RtspReconnectionMangerHandler, this);

        return true;
    }
    
    int RtspSourceBintr::ReconnectionManager()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_reconnectionManagerMutex);
        do
        {
            timeval currentTime;
            gettimeofday(&currentTime, NULL);
            
            uint stateResult(0);
            GstState currentState;
            
            if (!m_connectionData.is_in_reconnect or m_reconnectionFailed or 
                (currentTime.tv_sec - m_reconnectionStartTime.tv_sec) > m_connectionData.timeout)
            {
                // set the reset-state,
                if (!m_connectionData.is_in_reconnect)
                {
                    m_connectionData.is_connected = false;
                    m_connectionData.retries = 0;
                    m_connectionData.is_in_reconnect = true;
                }
                // if the previous attempt failed
                else if (m_reconnectionFailed == true)
                {
                    m_reconnectionSleep-=1;
                    if (m_reconnectionSleep)
                    {
                        LOG_INFO("Sleeping after failed connection");
                        return true;
                    }
                    m_reconnectionFailed = false;    
                }
                m_connectionData.retries++;

                LOG_INFO("Resetting RTSP Source '" << GetName() 
                    << "' with retry count = " << m_connectionData.retries);
                
                m_reconnectionStartTime = currentTime;

                if (SetState(GST_STATE_NULL, 0) != GST_STATE_CHANGE_SUCCESS)
                {
                    LOG_ERROR("Failed to set RTSP Source '" << GetName() << "' to GST_STATE_NULL");
                    return false;
                }
                // update the internal state variable to notify all client listeners 
                SetCurrentState(GST_STATE_NULL);
                return true;
            }
            else
            {   
                // Waiting for the Source to reconnect, check the state again
                stateResult = GetState(currentState, GST_SECOND);
            }
                
            // update the internal state variable to notify all client listeners 
            SetCurrentState(currentState);
            switch (stateResult) 
            {
                case GST_STATE_CHANGE_NO_PREROLL:
                    LOG_INFO("RTSP Source '" << GetName() 
                        << "' returned GST_STATE_CHANGE_NO_PREROLL");
                    // fall through ... do not break
                case GST_STATE_CHANGE_SUCCESS:
                    if (currentState == GST_STATE_NULL)
                    {
                        // synchronize the source's state with the Pipleine's
                        SyncStateWithParent(currentState, 1);
                        return true;
                    }
                    if (currentState == GST_STATE_PLAYING)
                    {
                        LOG_INFO("Re-connection complete for RTSP Source'" << GetName() << "'");
                        m_connectionData.is_in_reconnect = false;

                        // update the current buffer timestamp to the current reset time
                        m_TimestampPph->SetTime(currentTime);
                        m_reconnectionManagerTimerId = 0;
                        return false;
                    }
                    
                    // If state change completed succesfully, but not yet playing, set explicitely.
                    SetState(GST_STATE_PLAYING, 
                        DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND);
                    break;
                    
                case GST_STATE_CHANGE_ASYNC:
                    LOG_INFO("State change will complete asynchronously for RTSP Source '" 
                        << GetName() << "'");
                    break;

                case GST_STATE_CHANGE_FAILURE:
                    LOG_ERROR("FAILURE occured when trying to sync state for RTSP Source '" 
                        << GetName() << "'");
                    m_reconnectionFailed = true;
                    m_reconnectionSleep = m_connectionData.sleep;
                    return true;

                default:
                    LOG_ERROR("Unknown 'state change result' when trying to sync state for RTSP Source '" 
                        << GetName() << "'");
                    return true;
            }
        }while(true);
    }
    
    GstState RtspSourceBintr::GetCurrentState()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_stateChangeMutex);
        
        LOG_INFO("Returning state " 
            << gst_element_state_get_name((GstState)m_currentState) << 
            " for RtspSourceBintr '" << GetName() << "'");

        return m_currentState;
    }

    void RtspSourceBintr::SetCurrentState(GstState newState)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_stateChangeMutex);

        if (newState != m_currentState)
        {
            LOG_INFO("Changing state from " << 
                gst_element_state_get_name((GstState)m_currentState) << 
                " to " << gst_element_state_get_name((GstState)newState) 
                << " for RtspSourceBintr '" << GetName() << "'");
            
            m_previousState = m_currentState;
            m_currentState = newState;

            struct timeval currentTime;
            gettimeofday(&currentTime, NULL);
            
            if ((m_previousState == GST_STATE_PLAYING) and (m_currentState == GST_STATE_NULL))
            {
                m_connectionData.is_connected = false;
                m_connectionData.last_disconnected = currentTime.tv_sec;
            }
            if (m_currentState == GST_STATE_PLAYING)
            {
                m_connectionData.is_connected = true;
                
                // if first time is empty, this is the first since Pipeline play or stats clear.
                if(!m_connectionData.first_connected)
                {
                    m_connectionData.first_connected = currentTime.tv_sec;
                }
                m_connectionData.last_connected = currentTime.tv_sec;
                m_connectionData.count++;
            }                    
            
            if (m_stateChangeListeners.size())
            {
                std::shared_ptr<DslStateChange> pStateChange = 
                    std::shared_ptr<DslStateChange>(new DslStateChange(m_previousState, m_currentState));
                    
                m_stateChanges.push(pStateChange);
                
                // start the asynchronous notification timer if not currently running
                if (!m_listenerNotifierTimerId)
                {
                    m_listenerNotifierTimerId = g_timeout_add(1, RtspListenerNotificationHandler, this);
                }
            }
        }
    }
    
    int RtspSourceBintr::NotifyClientListeners()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_stateChangeMutex);
        
        while (m_stateChanges.size())
        {
            std::shared_ptr<DslStateChange> pStateChange = m_stateChanges.front();
            m_stateChanges.pop();
            
            // iterate through the map of state-change-listeners calling each
            for(auto const& imap: m_stateChangeListeners)
            {
                try
                {
                    imap.first((uint)pStateChange->m_previousState, 
                        (uint)pStateChange->m_newState, imap.second);
                }
                catch(...)
                {
                    LOG_ERROR("RTSP Source '" << GetName() 
                        << "' threw exception calling Client State-Change-Lister");
                }
            }
            
        }
        // clear the timer id and return false to self remove
        m_listenerNotifierTimerId = 0;
        return false;
    }
    
    // --------------------------------------------------------------------------------------

    static int ImageSourceDisplayTimeoutHandler(gpointer pSource)
    {
        return static_cast<ImageStreamSourceBintr*>(pSource)->
            HandleDisplayTimeout();
    }
    
    static void UriSourceElementOnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource)
    {
        static_cast<UriSourceBintr*>(pSource)->HandleSourceElementOnPadAdded(pBin, pPad);
    }
    
    static boolean RtspSourceSelectStreamCB(GstElement *pBin, uint num, GstCaps *caps,
        gpointer pSource)
    {
        return static_cast<RtspSourceBintr*>(pSource)->HandleSelectStream(pBin, num, caps);
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

    static int RtspStreamManagerHandler(gpointer pSource)
    {
        return static_cast<RtspSourceBintr*>(pSource)->
            StreamManager();
    }

    static int RtspReconnectionMangerHandler(gpointer pSource)
    {
        return static_cast<RtspSourceBintr*>(pSource)->
            ReconnectionManager();
    }

    static int RtspListenerNotificationHandler(gpointer pSource)
    {
        return static_cast<RtspSourceBintr*>(pSource)->
            NotifyClientListeners();
    }
    
} // SDL namespace
