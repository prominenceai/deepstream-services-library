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

#define N_DECODE_SURFACES 16
#define N_EXTRA_SURFACES 1

namespace DSL
{
    SourceBintr::SourceBintr(const char* name)
        : Bintr(name)
        , m_sourceId(-1)
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
    
    bool SourceBintr::AddToParent(DSL_NODETR_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' Source to the Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            AddSourceBintr(std::dynamic_pointer_cast<SourceBintr>(shared_from_this()));
    }

    bool SourceBintr::IsParent(DSL_NODETR_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // check if 'this' Source is child of Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            IsSourceBintrChild(std::dynamic_pointer_cast<SourceBintr>(shared_from_this()));
    }

    bool SourceBintr::RemoveFromParent(DSL_NODETR_PTR pParentBintr)
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

    int SourceBintr::GetSourceId()
    {
        LOG_FUNC();
        
        return m_sourceId;
    }

    void SourceBintr::SetSourceId(int id)
    {
        LOG_FUNC();

        if (!m_pSourceElement)
        {
            LOG_ERROR("Source Element for SourceBintr '" << GetName() << "' has not been instantiated");
            throw;
        }
        m_sourceId = id;
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

        std::string sinkPadName = "sink_" + std::to_string(m_sourceId);
        
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

        // If we're currently linked to the 
        if (!IsLinkedToSink())
        {
            LOG_ERROR("SourceBintr '" << GetName() << "' is not in a Linked state");
            return false;
        }

        std::string sinkPadName = "sink_" + std::to_string(m_sourceId);

        LOG_INFO("Unlinking and releasing request Sink Pad for StreamMux " << m_pSink->GetName());

        gst_pad_unlink(m_pGstStaticSourcePad, m_pGstRequestedSinkPads[sinkPadName]);
        gst_element_release_request_pad(GetSink()->GetGstElement(), m_pGstRequestedSinkPads[sinkPadName]);

        m_pGstRequestedSinkPads.erase(sinkPadName);
        
        return Nodetr::UnlinkFromSink();
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
        m_pSourceElement->SetAttribute("maxperf", TRUE);

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

    FileSourceBintr::FileSourceBintr(const char* name, const char* filePath, const char* parserName)
        : SourceBintr(name)
        , m_parserName(parserName)
    {
        LOG_FUNC();

        std::ifstream streamFile(filePath);
        if (!streamFile.good())
        {
            LOG_ERROR("Source file '" << filePath << "' Not found");
            throw;
        }
        
        if ((m_parserName.find("parse") == std::string::npos) or
            (!gst_element_factory_find(parserName)))
        {
            LOG_ERROR("Parser name '" << parserName << "' is invalid or Parser is not installed");
            throw;
        }
        
        // File source, not live - setup full path
        m_isLive = FALSE;
        char absolutePath[PATH_MAX+1];
        m_filePath = realpath(filePath, absolutePath);
        
        LOG_INFO("Source file path = " << m_filePath);

        m_pSourceElement = DSL_ELEMENT_NEW("filesrc", "file-source");
        m_pParser = DSL_ELEMENT_NEW(parserName, "h264-parser");
        m_pDecoder = DSL_ELEMENT_NEW("nvv4l2decoder", "v4l2-decoder");
        
        m_pSourceElement->SetAttribute("location", m_filePath.c_str());

        AddChild(m_pSourceElement);
        AddChild(m_pParser);
        AddChild(m_pDecoder);
        
        m_pDecoder->AddGhostPadToParent("src");
    }

    FileSourceBintr::~FileSourceBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }
    
    bool FileSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("FileSourceBintr '" << GetName() << "' is already in a linked state");
            return false;
        }
        m_pSourceElement->LinkToSink(m_pParser);
        m_pParser->LinkToSink(m_pDecoder);
        m_isLinked = true;
        
        return true;
    }

    void FileSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("FileSourceBintr '" << GetName() << "' is not in a linked state");
            return;
        }
        m_pSourceElement->UnlinkFromSink();
        m_pParser->UnlinkFromSink();
        m_isLinked = false;
    }
    
    const char* FileSourceBintr::GetFilePath()
    {
        LOG_FUNC();
        
        return m_filePath.c_str();
    }
    
    const char* FileSourceBintr::GetParserName()
    {
        LOG_FUNC();
        
        return m_parserName.c_str();
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
        if (!m_isLive)
        {
            std::ifstream streamUriFile(uri);
            if (!streamUriFile.good())
            {
                LOG_ERROR("URI '" << uri << "' Not found");
                throw;
            }        
            
            // File source, not live - setup full path
            char absolutePath[PATH_MAX+1];
            m_uri = realpath(uri, absolutePath);
            m_uri.insert(0, "file:");
        }
        else
        {
            m_uri = uri;
        }
        
        LOG_INFO("URI Path = " << m_uri);
        std::string sourceElementName = "src-element" + GetName();
        m_pSourceElement = DSL_ELEMENT_NEW(factoryName, sourceElementName.c_str());
        
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
            g_object_set(pObject, "source-id", m_sourceId, NULL);
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
#ifdef __aarch64__
            g_object_set(pObject, "enable-max-performance", TRUE, NULL);
            g_object_set(pObject, "bufapi-version", TRUE, NULL);
#else
            g_object_set(pObject, "gpu-id", pDecodeSourceBintr->gpuId, NULL);
            g_object_set(G_OBJECT(pObject), "cudadec-memtype", m_cudadecMemtype, NULL);
#endif
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
        Pause();
        
        gboolean retval = gst_element_seek(GetGstElement(), 1.0, GST_FORMAT_TIME,
            (GstSeekFlags)(GST_SEEK_FLAG_KEY_UNIT | GST_SEEK_FLAG_FLUSH),
            GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_NONE, GST_CLOCK_TIME_NONE);

        if (!retval)
        {
            LOG_WARN("Failure to seek");
        }

        Play();
        return false;
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
        std::string padForSourceQueueName = "padForSourceQueue_" + std::to_string(m_sourceId);

        m_pGstRequestedSourcePads[padForSourceQueueName] = pGstRequestedSourcePad;
        m_pSourceQueue->LinkToSource(m_pTee);

        pGstRequestedSourcePad = gst_element_request_pad(m_pTee->GetGstElement(), pPadTemplate, NULL, NULL);
        if (!pGstRequestedSourcePad)
        {
            LOG_ERROR("Failed to get Tee Pad for PipelineSinksBintr '" << GetName() <<"'");
            return false;
        }
        std::string padForFakeSinkQueueName = "padForFakeSinkQueue_" + std::to_string(m_sourceId);

        m_pGstRequestedSourcePads[padForFakeSinkQueueName] = pGstRequestedSourcePad;
        m_pFakeSinkQueue->LinkToSource(m_pTee);
    
        m_pFakeSinkQueue->LinkToSink(m_pFakeSink);
        m_isLinked = true;

        return true;
    }

    void UriSourceBintr::UnlinkAll()
    {
        LOG_FUNC();
    
        if (!m_isLinked)
        {
            LOG_ERROR("CsiSourceBintr '" << GetName() << "' is not in a linked state");
            return;
        }
        // If the Source Queue is Linked back with Tee
        if (m_pSourceQueue->IsLinkedToSource())
        {
            m_pSourceQueue->UnlinkFromSource();
        }
        if (m_pSourceQueue->IsLinkedToSink())
        {
            m_pSourceQueue->UnlinkFromSink();
        }
        // If the Source Queue is Linked back with Tee
        if (m_pFakeSinkQueue->IsLinkedToSource())
        {
            m_pFakeSinkQueue->UnlinkFromSource();
        }
        if (m_pFakeSinkQueue->IsLinkedToSink())
        {
            m_pFakeSinkQueue->UnlinkFromSink();
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

    //*********************************************************************************
    
    RtspSourceBintr::RtspSourceBintr(const char* name, const char* uri, uint protocol,
        uint cudadecMemType, uint intraDecode, uint dropFrameInterval)
        : DecodeSourceBintr(name, "rtspsrc", uri, true, cudadecMemType, intraDecode, dropFrameInterval)
        , m_rtpProtocols(protocol)
    {
        LOG_FUNC();
        
        // New RTSP Specific Elementrs for this Source
        m_pDepayload = DSL_ELEMENT_NEW("rtph264depay", "src-depayload");
        m_pDecodeQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "decode-queue");
        m_pDecodeBin = DSL_ELEMENT_NEW("decodebin", "decode-bin");
        m_pSourceQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "src-queue");

        m_pSourceElement->SetAttribute("location", m_uri.c_str());
        m_pSourceElement->SetAttribute("latency", m_latency);
        m_pSourceElement->SetAttribute("drop-on-latency", true);
        m_pSourceElement->SetAttribute("protocols", m_rtpProtocols);

        // Connect RTSP Source Setup Callbacks
        g_signal_connect(m_pSourceElement->GetGObject(), "pad-added", 
            G_CALLBACK(RtspSourceElementOnPadAddedCB), this);

        // Connect Decode Setup Callbacks
        g_signal_connect(m_pDecodeBin->GetGObject(), "pad-added", 
            G_CALLBACK(RtspDecodeElementOnPadAddedCB), this);
        g_signal_connect(m_pDecodeBin->GetGObject(), "child-added", 
            G_CALLBACK(OnChildAddedCB), this);

//        g_object_set_data(G_OBJECT(m_pSourceElement->GetGObject()), "source", this);

        AddChild(m_pDepayload);
        AddChild(m_pDecodeQueue);
        AddChild(m_pDecodeBin);
        AddChild(m_pSourceQueue);
        
        // Source Ghost Pad for Source Queue
        m_pSourceQueue->AddGhostPadToParent("src");
    }

    RtspSourceBintr::~RtspSourceBintr()
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            UnlinkAll();
        }
    }
    
    bool RtspSourceBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("RtspSourceBintr '" << GetName() << "' is already in a linked state");
            return false;
        }

        m_pDepayload->LinkToSink(m_pDecodeQueue);
        m_pDecodeQueue->LinkToSink(m_pDecodeBin);
        m_pDecodeBin->LinkToSink(m_pDecodeQueue);
        m_isLinked = true;
        
        return true;
    }

    void RtspSourceBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("CsiSourceBintr '" << GetName() << "' is not in a linked state");
            return;
        }
        m_pDepayload->UnlinkFromSink();
        m_pDecodeQueue->UnlinkFromSink();
        m_pDecodeBin->UnlinkFromSink();
        m_isLinked = false;
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
            m_pGstStaticSinkPad = gst_element_get_static_pad(m_pDepayload->GetGstElement(), "sink");
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
    
    
    static void UriSourceElementOnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource)
    {
        static_cast<UriSourceBintr*>(pSource)->HandleSourceElementOnPadAdded(pBin, pPad);
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

} // SDL namespace