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

        // Single Stream Muxer element for all Sources 
        m_pStreamMux = MakeElement(NVDS_ELEM_STREAM_MUX, "stream_muxer", LINK_TRUE);

        // Each Source added will be linked to the Stream Muxer
        
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

        m_pChildBintrs.push_back(pChildBintr);
                        
        if (!gst_bin_add(GST_BIN(m_pBin), pChildBintr->m_pBin))
        {
            LOG_ERROR("Failed to add '" << pChildBintr->m_name 
                << "' to " << m_name << "'");
            throw;
        }

        // Get the static source pad - from the Source component
        // being added - to link to Streammux element
        StaticPadtr SourcePadtr(pChildBintr->m_pBin, "src");
        
        // Retrieve the sink pad - from the Streammux element - 
        // to link to the Source component being added
        RequestPadtr SinkPadtr(m_pStreamMux, "sink_0");
     
        if (gst_pad_link(SourcePadtr.m_pPad, SinkPadtr.m_pPad) != GST_PAD_LINK_OK)
        {
            LOG_ERROR("Failed to link '" << pChildBintr->m_name 
                << "' to Stream Muxer" << m_name << "'");
            throw;
        }
        
        LOG_INFO("Source '" << pChildBintr->m_name << "' linked to Stream Muxer");
    }
    
    void SourcesBintr::AddSourceGhostPad()
    {
        LOG_FUNC();
        
        // get Source pad for Stream Muxer element
        StaticPadtr SourcePadtr(m_pStreamMux, "src");

        // create a new ghost pad with Source pad and add to this Bintr's bin
        if (!gst_element_add_pad(m_pBin, gst_ghost_pad_new("src", SourcePadtr.m_pPad)))
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

    
    CsiSourceBintr::CsiSourceBintr(const char* source, 
        guint width, guint height, guint fps_n, guint fps_d)
        : Bintr(source)
        , m_isLive(TRUE)
        , m_width(width)
        , m_height(height)
        , m_fps_n(fps_n)
        , m_fps_d(fps_d)
        , m_latency(100)
        , m_numDecodeSurfaces(N_DECODE_SURFACES)
        , m_numExtraSurfaces(N_EXTRA_SURFACES)
        , m_pSourceElement(NULL)
        , m_pCapsFilter(NULL)
    {
        LOG_FUNC();
              
        // Create Source Element and Caps filter - Order is specific
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
        
        // Src Ghost Pad only
        AddSourceGhostPad();
    }

    CsiSourceBintr::~CsiSourceBintr()
    {
        LOG_FUNC();

    }

    void CsiSourceBintr::AddToParent(std::shared_ptr<Bintr> pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' Source to the Parent Pipeline 
        std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)-> \
            AddCsiSourceBintr(shared_from_this());
    }

    UriSourceBintr::UriSourceBintr(const char* source, const char* uri,
        guint width, guint height, guint fps_n, guint fps_d)
        : Bintr(source)
        , m_isLive(TRUE)
        , m_width(width)
        , m_height(height)
        , m_fps_n(fps_n)
        , m_fps_d(fps_d)
        , m_latency(100)
        , m_numDecodeSurfaces(N_DECODE_SURFACES)
        , m_numExtraSurfaces(N_EXTRA_SURFACES)
        , m_pSourceElement(NULL)
        , m_pCapsFilter(NULL)
    {
        LOG_FUNC();
        
        m_uri = uri;
              
        // Create Source Element and Caps filter - Order is specific
        m_pSourceElement = MakeElement(NVDS_ELEM_SRC_URI, "src_elem", LINK_TRUE);

        g_object_set(G_OBJECT(m_pSourceElement), "uri", config->uri, NULL);
        g_signal_connect(G_OBJECT(m_pSourceElement), "pad-added", 
            G_CALLBACK(cb_newpad), bin);
        g_signal_connect(G_OBJECT(m_pSourceElement), "child-added", 
            G_CALLBACK(decodebin_child_added), bin);
        g_signal_connect (G_OBJECT (bin->src_elem), "source-setup",
            G_CALLBACK(cb_sourcesetup), bin);
        
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
        
        // Src Ghost Pad only
        AddSourceGhostPad();
    }

    UriSourceBintr::~UriSourceBintr()
    {
        LOG_FUNC();
    }

    void UriSourceBintr::AddToParent(std::shared_ptr<Bintr> pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' Source to the Parent Pipeline 
        std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)-> \
            AddUriSourceBintr(shared_from_this());
    }
    
    static void newpad(GstElement* pBin, GstPad* pPad, gpointer pSource)
    {
        LOG_FUNC();
        
        std:shared_ptr<UriSourceBintr> pUriSourceBintr = 
            std::dynamic_pointer_cast<UriSourceBintr>(pSource);
        
        // get Sink pad for first child element in the ordered list
        StaticPadtr sinkPadtr(pUriSourceBintr->m_pTee, "sink");
        
        if (gst_pad_link(pPad, sinkPadtr->m_pPad) != GST_PAD_LINK_OK) 
        {
            LOG_ERROR("Failed to link decodebin to pipeline");
        }
        else
        {
            LOG_INFO("Decodebin linked to pipeline");
        }
    }
    
}