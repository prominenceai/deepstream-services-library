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
    SourcesBintr::SourcesBintr(const char* name)
        : Bintr(name)
    {
        LOG_FUNC();

        g_object_set(m_pBin, "message-forward", TRUE, NULL);
  
        // Single Stream Muxer element for all Sources 
        m_pStreamMux = MakeElement(NVDS_ELEM_STREAM_MUX, "stream_muxer", LINK_TRUE);

        // Each Source will be linked to the Stream Muxer later, on Source add
        
        // Setup Src Ghost Pad for Stream Muxer element 
        AddSourceGhostPad();
    }
    
    SourcesBintr::~SourcesBintr()
    {
        LOG_FUNC();
    }
     
    void SourcesBintr::AddChild(std::shared_ptr<Bintr> pChildBintr)
    {
        LOG_FUNC();
        
        pChildBintr->m_pParentBintr = 
            std::dynamic_pointer_cast<Bintr>(shared_from_this());

        m_pChildBintrs[pChildBintr->m_name] = pChildBintr;

        // set the source ID based on the new count of sources
//        std::dynamic_pointer_cast<SourceBintr>(pChildBintr)->
//            m_sourceId = m_pChildBintrs.size();
                                
        if (!gst_bin_add(GST_BIN(m_pBin), pChildBintr->m_pBin))
        {
            LOG_ERROR("Failed to add '" << pChildBintr->m_name 
                << "' to " << m_name << "'");
            throw;
        }

        // Retrieve the sink pad - from the Streammux element - 
        // to link to the Source component being added
        std::shared_ptr<RequestPadtr> pSinkPadtr = 
            std::shared_ptr<RequestPadtr>(new RequestPadtr(m_pStreamMux, "sink_0"));
        
        std::dynamic_pointer_cast<SourceBintr>(pChildBintr)->
            m_pStaticSourcePadtr->LinkTo(pSinkPadtr);
        
        LOG_INFO("Source '" << pChildBintr->m_name << "' linked to Stream Muxer");
    }

    void SourcesBintr::RemoveChild(std::shared_ptr<Bintr> pChildBintr)
    {
        LOG_FUNC();

        std::dynamic_pointer_cast<SourceBintr>(pChildBintr)->
            m_pStaticSourcePadtr->Unlink();
        
        Bintr::RemoveChild(pChildBintr);
    }

    void SourcesBintr::AddSourceGhostPad()
    {
        LOG_FUNC();
        
        // get Source pad for Stream Muxer element
        StaticPadtr sourcePadtr(m_pStreamMux, "src");

        // create a new ghost pad with Source pad and add to this Bintr's bin
        if (!gst_element_add_pad(m_pBin, gst_ghost_pad_new("src", sourcePadtr.m_pPad)))
        {
            LOG_ERROR("Failed to add Source Pad for '" << m_name);
            throw;
        }
        LOG_INFO("Source ghost pad added to Sources' Stream Muxer"); 
    }

    void SourcesBintr::SetStreamMuxProperties(gboolean areSourcesLive, 
        guint batchSize, guint batchTimeout, guint width, guint height)
    {
        m_areSourcesLive = areSourcesLive;
        m_batchSize = batchSize;
        m_batchTimeout = batchTimeout;
        m_streamMuxWidth = width;
        m_streamMuxHeight = height;
        m_enablePadding = FALSE;
        
        g_object_set(G_OBJECT(m_pStreamMux), "gpu-id", m_gpuId, NULL);
        g_object_set(G_OBJECT(m_pStreamMux), "nvbuf-memory-type", m_nvbufMemoryType, NULL);
        g_object_set(G_OBJECT(m_pStreamMux), "live-source", m_areSourcesLive, NULL);
        g_object_set(G_OBJECT(m_pStreamMux), "batched-push-timeout", m_batchTimeout, NULL);

        if ((gboolean)m_batchSize)
        {
            g_object_set(G_OBJECT(m_pStreamMux), "batch-size", m_batchSize, NULL);
        }

        g_object_set(G_OBJECT(m_pStreamMux), "enable-padding", m_enablePadding, NULL);

        if (m_streamMuxWidth && m_streamMuxHeight)
        {
            g_object_set(G_OBJECT(m_pStreamMux), "width", m_streamMuxWidth, NULL);
            g_object_set(G_OBJECT(m_pStreamMux), "height", m_streamMuxHeight, NULL);
        }
        LOG_INFO("Sources' Stream Muxer properties updated"); 
    }

    SourceBintr::SourceBintr(const char* source, guint width, guint height, 
        guint fps_n, guint fps_d)
        : Bintr(source)
        , m_sourceId((guint)-1)
        , m_isLive(TRUE)
        , m_width(width)
        , m_height(height)
        , m_fps_n(fps_n)
        , m_fps_d(fps_d)
        , m_latency(100)
        , m_numDecodeSurfaces(N_DECODE_SURFACES)
        , m_numExtraSurfaces(N_EXTRA_SURFACES)
        , m_pSourceElement(NULL)
    {
        LOG_FUNC();
    }

    SourceBintr::SourceBintr(const char* source)
        : Bintr(source)
        , m_sourceId((guint)-1)
        , m_isLive(TRUE)
        , m_width(0)
        , m_height(0)
        , m_fps_n(0)
        , m_fps_d(0)
        , m_latency(100)
        , m_numDecodeSurfaces(N_DECODE_SURFACES)
        , m_numExtraSurfaces(N_EXTRA_SURFACES)
        , m_pSourceElement(NULL)
    {
        LOG_FUNC();
    }
    
    SourceBintr::~SourceBintr()
    {
        LOG_FUNC();
    }
    
    void SourceBintr::AddToParent(std::shared_ptr<Bintr> pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' Source to the Parent Pipeline 
        std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            AddSourceBintr(shared_from_this());
    }

    void SourceBintr::RemoveFromParent(std::shared_ptr<Bintr> pParentBintr)
    {
        LOG_FUNC();
        
        // remove 'this' Source from the Parent Pipeline 
        std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            RemoveSourceBintr(shared_from_this());
    }


    CsiSourceBintr::CsiSourceBintr(const char* source, 
        guint width, guint height, guint fps_n, guint fps_d)
        : SourceBintr(source, width, height, fps_n, fps_d)
        , m_pCapsFilter(NULL)
    {
        LOG_FUNC();

        // Create Source Element and Caps filter - Elements are linked in the order added
        m_pSourceElement = MakeElement(NVDS_ELEM_SRC_CAMERA_CSI, "src_elem", LINK_TRUE);
        m_pCapsFilter = MakeElement(NVDS_ELEM_CAPS_FILTER, "src_cap_filter", LINK_TRUE);

        g_object_set(G_OBJECT(m_pSourceElement), "bufapi-version", TRUE, NULL);
        g_object_set(G_OBJECT(m_pSourceElement), "maxperf", TRUE, NULL);
        g_object_set(G_OBJECT(m_pSourceElement), "sensor-id", 0, NULL);

        GstCaps * pCaps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12",
            "width", G_TYPE_INT, m_width, "height", G_TYPE_INT, m_height, 
            "framerate", GST_TYPE_FRACTION, m_fps_n, m_fps_d, NULL);
        if (!pCaps)
        {
            LOG_ERROR("Failed to create new Simple Capabilities for '" << source << "'");
            throw;  
        }

        GstCapsFeatures *feature = NULL;
        feature = gst_caps_features_new("memory:NVMM", NULL);

        gst_caps_set_features(pCaps, 0, feature);
        g_object_set(G_OBJECT(m_pCapsFilter), "caps", pCaps, NULL);
        
        gst_caps_unref(pCaps);        
        
        // Add Ghost Pad and create Static Padtr to link to StreamMuxer
        AddSourceGhostPad();

        m_pStaticSourcePadtr = std::shared_ptr<StaticPadtr>(new StaticPadtr(m_pBin, "src"));
    }

    CsiSourceBintr::~CsiSourceBintr()
    {
        LOG_FUNC();

    }

    UriSourceBintr::UriSourceBintr(const char* source, const char* uri,
        guint cudadecMemType, guint intraDecode)
        : SourceBintr(source)
        , m_uriString(uri)
        , m_cudadecMemtype(cudadecMemType)
        , m_intraDecode(intraDecode)
        , m_accumulatedBase(0)
        , m_prevAccumulatedBase(0)
        , m_pSourceQueue(NULL)
        , m_pTee(NULL)
        , m_pFakeSink(NULL)
        , m_pFakeSinkQueue(NULL)
    {
        LOG_FUNC();
        
        m_uriString = uri;
        
        m_isLive = FALSE;
              
        // Create Source Element - without linking at this time.
        m_pSourceElement = MakeElement(NVDS_ELEM_SRC_URI, "src_elem", LINK_FALSE);
        g_object_set(G_OBJECT(m_pSourceElement), "sensor-id", 0, NULL);
        
        g_object_set(G_OBJECT(m_pSourceElement), "uri", (gchar*)uri, NULL);

        g_signal_connect(G_OBJECT(m_pSourceElement), "pad-added", 
            G_CALLBACK(OnPadAddedCB), this);
        g_signal_connect(G_OBJECT(m_pSourceElement), "child-added", 
            G_CALLBACK(OnChildAddedCB), this);
        g_signal_connect(G_OBJECT(m_pSourceElement), "source-setup",
            G_CALLBACK(OnSourceSetupCB), this);

        // Create a Queue for the URI source
        m_pSourceQueue = MakeElement(NVDS_ELEM_QUEUE, "src-queue", LINK_TRUE);
        g_object_set_data(G_OBJECT(m_pSourceQueue), "source", this);
        
        // Source Ghost Pad for Source Queue
        AddSourceGhostPad();
        
        m_pTee = MakeElement(NVDS_ELEM_TEE, "tee", LINK_FALSE);

        m_pFakeSinkQueue = MakeElement(NVDS_ELEM_QUEUE, "fake-sink-queue", LINK_FALSE);
        m_pFakeSink = MakeElement(NVDS_ELEM_SINK_FAKESINK, "fake-sink", LINK_FALSE);

        if (!gst_element_link(m_pFakeSinkQueue, m_pFakeSink))
        {
            LOG_ERROR("Failed to link Queue with Fake Sink for source '" << m_name << "'");
            throw;
        }

        GstPadTemplate* padtemplate = 
            gst_element_class_get_pad_template(GST_ELEMENT_GET_CLASS(m_pTee), "src_%u");
        if (!padtemplate)
        {
            LOG_ERROR("Failed to get Pad Template for '" << m_name << "'");
            throw;
        }
        
        // The TEE for this source is linked to both the "source queue" and "fake sink queue"
        
        // get a pad from the Tee element and link to the "source queuue"
        RequestPadtr teeSourcePadtr1(m_pTee, padtemplate, "src");
        StaticPadtr sourceQueueSinkPadtr(m_pSourceQueue, "sink");
//        teeSourcePadtr1.LinkTo(sourceQueueSinkPadtr);

        // get a second pad from the Tee element and link to the "fake sink queue"
        RequestPadtr teeSourcePadtr2(m_pTee, padtemplate, "src");
        StaticPadtr fakeSinkQueueSinkPadtr(m_pFakeSinkQueue, "sink");
//        teeSourcePadtr2.LinkTo(fakeSinkQueueSinkPadtr);

        g_object_set(G_OBJECT(m_pFakeSink), "sync", FALSE, "async", FALSE, NULL);
        
    }

    UriSourceBintr::~UriSourceBintr()
    {
        LOG_FUNC();
    }

    void UriSourceBintr::HandleOnPadAdded(GstElement* pBin, GstPad* pPad)
    {
        LOG_FUNC();

        // The "pad-added" callback will be called twice for each URI source,
        // once each for the decoded Audio and Video streams. Since we only 
        // want to link to the Video source pad, we need to know which of the
        // two pads/calls this is for.
        GstCaps* pCaps = gst_pad_query_caps(pPad, NULL);
        GstStructure* structure = gst_caps_get_structure(pCaps, 0);
        std::string name = gst_structure_get_name(structure);
        
        LOG_INFO("Caps structs name " << name);
        if (name.find("video") != std::string::npos)
        {
            // get a static Sink pad for this URI source bintr's Tee element
            StaticPadtr sinkPadtr(m_pTee, "sink");
            
            if (gst_pad_link(pPad, sinkPadtr.m_pPad) != GST_PAD_LINK_OK) 
            {
                LOG_ERROR("Failed to link decodebin to pipeline");
                throw;
            }
            
            // Update the cap memebers for this URI source bintr
            gst_structure_get_uint(structure, "width", &m_width);
            gst_structure_get_uint(structure, "height", &m_height);
            gst_structure_get_fraction(structure, "framerate", (gint*)&m_fps_n, (gint*)&m_fps_d);
            
            LOG_INFO("Video decode linked for URI source '" << m_name << "'");
        }
    }

    void UriSourceBintr::HandleOnChildAdded(GstChildProxy* pChildProxy, GObject* pObject,
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

        else if ((strName.find("omx") != std::string::npos) && m_intraDecode)
        {
            g_object_set(pObject, "skip-frames", 2, NULL);
            g_object_set(pObject, "disable-dvfs", TRUE, NULL);
        }

        else if (strName.find("nvjpegdec") != std::string::npos)
        {
            g_object_set(pObject, "DeepStream", TRUE, NULL);
        }

        else if ((strName.find("nvv4l2decoder") != std::string::npos) && m_intraDecode)
        {
            g_object_set (pObject, "skip-frames", 2, NULL);
#ifdef __aarch64__
            g_object_set(pObject, "enable-max-performance", TRUE, NULL);
            g_object_set(pObject, "bufapi-version", TRUE, NULL);
#else
            g_object_set(pObject, "gpu-id", pUriSourceBintr->gpuId, NULL);
            g_object_set(G_OBJECT(pObject), "cudadec-memtype", m_cudadecMemtype, NULL);
#endif
            g_object_set(pObject, "drop-frame-interval", m_dropFrameInterval, NULL);
            g_object_set(pObject, "num-extra-surfaces", m_numExtraSurfaces, NULL);

            // if the source is from file, then setup Stream buffer probe function
            // to handle the stream restart/loop on GST_EVENT_EOS.
            if (!m_isLive)
            {
                GstPadProbeType mask = (GstPadProbeType) 
                    (GST_PAD_PROBE_TYPE_EVENT_BOTH |
                    GST_PAD_PROBE_TYPE_EVENT_FLUSH | 
                    GST_PAD_PROBE_TYPE_BUFFER);
                    
                StaticPadtr sinkPadtr(GST_ELEMENT(pObject), "sink");
                
                m_bufferProbeId = sinkPadtr.AddPad(mask, StreamBufferRestartProbCB, this);
            }
        }
    }
    
    GstPadProbeReturn UriSourceBintr::HandleStreamBufferRestart(GstPad* pPad, GstPadProbeInfo* pInfo)
    {
        LOG_FUNC();
        
        GstEvent* event = GST_EVENT(pInfo->data);

        if (pInfo->type & GST_PAD_PROBE_TYPE_BUFFER)
        {
            GST_BUFFER_PTS(GST_BUFFER(pInfo->data)) += m_prevAccumulatedBase;
        }
        
        else if (pInfo->type & GST_PAD_PROBE_TYPE_EVENT_BOTH)
        {
            if (GST_EVENT_TYPE(event) == GST_EVENT_EOS)
            {
                g_timeout_add(1, StreamBufferSeekCB, this);
            }
            else if (GST_EVENT_TYPE(event) == GST_EVENT_SEGMENT)
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

    void UriSourceBintr::HandleOnSourceSetup(GstElement* pObject, GstElement* arg0)
    {
        if (g_object_class_find_property(G_OBJECT_GET_CLASS(arg0), "latency")) 
        {
            g_object_set(G_OBJECT(arg0), "latency", "cb_sourcesetup set %d latency\n", NULL);
        }
    }
    
    gboolean UriSourceBintr::HandleStreamBufferSeek()
    {
        gst_element_set_state(m_pBin, GST_STATE_PAUSED);

        gboolean retval = gst_element_seek(m_pBin, 1.0, GST_FORMAT_TIME,
            (GstSeekFlags)(GST_SEEK_FLAG_KEY_UNIT | GST_SEEK_FLAG_FLUSH),
            GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_NONE, GST_CLOCK_TIME_NONE);

        if (!retval)
        {
            LOG_WARN("Failure to seek");
        }

        gst_element_set_state(m_pBin, GST_STATE_PLAYING);
    }
    
    
    static void OnPadAddedCB(GstElement* pBin, GstPad* pPad, gpointer pSource)
    {
        static_cast<UriSourceBintr*>(pSource)->HandleOnPadAdded(pBin, pPad);
    }
    
    static void OnChildAddedCB(GstChildProxy* pChildProxy, GObject* pObject,
        gchar* name, gpointer pSource)
    {
        static_cast<UriSourceBintr*>(pSource)->HandleOnChildAdded(pChildProxy, pObject, name);
    }
    
    static void OnSourceSetupCB(GstElement* pObject, GstElement* arg0, 
        gpointer pSource)
    {
        static_cast<UriSourceBintr*>(pSource)->HandleOnSourceSetup(pObject, arg0);
    }
    
    static GstPadProbeReturn StreamBufferRestartProbCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pSource)
    {
        return static_cast<UriSourceBintr*>(pSource)->
            HandleStreamBufferRestart(pPad, pInfo);
    }

    static gboolean StreamBufferSeekCB(gpointer pSource)
    {
        return static_cast<UriSourceBintr*>(pSource)->HandleStreamBufferSeek();
    }
} // SDL namespace