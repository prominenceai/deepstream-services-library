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

#include "Dsd.h"
#include "DsdOsdBintr.h"

namespace DSD
{
    OsdBintr::OsdBintr(const std::string& osd, guint displayId, guint overlayId,
        guint offsetX, guint offsetY, guint width, guint height)
        : Bintr(osd)
        , m_displayId(displayId)
        , m_overlayId(overlayId)
        , m_offsetX(offsetX)
        , m_offsetY(offsetY)
        , m_width(width)
        , m_height(height)
        , m_pQueue(NULL)
        , m_pTee(NULL)
        , m_pSink(NULL)
    {
        LOG_FUNC();
        
        m_pBin = gst_bin_new((gchar*)osd.c_str());
        if (!m_pBin)
        {
            LOG_ERROR("Failed to create new OSD bin for '" << osd << "'");
            throw;  
        }

        m_pVidConv = gst_element_factory_make(NVDS_ELEM_VIDEO_CONV, "osd_conv");
        if (!m_pVidConv)
        {
            LOG_ERROR("Failed to create new Video Conv for OSD '" << osd << "'");
            throw;  
        }

        m_pQueue = gst_element_factory_make(NVDS_ELEM_QUEUE, "osd_queue");
        if (!m_pQueue)
        {
            LOG_ERROR("Failed to create new Queue for OSD '" << osd << "'");
            throw;  
        }
        
        m_pCapsFilter = gst_element_factory_make(NVDS_ELEM_CAPS_FILTER, "osd_caps");
        if (!m_pCapsFilter)
        {
            LOG_ERROR("Failed to create new Caps for OSD '" << osd << "'");
            throw;  
        }
        
        m_pConvQueue = gst_element_factory_make(NVDS_ELEM_QUEUE, "osd_conv_queue");
        if (!m_pConvQueue)
        {
            LOG_ERROR("Failed to create new  for OSD '" << osd << "'");
            throw;  
        }

        GstCaps *caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGBA", NULL);

        GstCapsFeatures *feature = gst_caps_features_new(MEMORY_FEATURES, NULL);
        gst_caps_set_features(caps, 0, feature);
        g_object_set(G_OBJECT(m_pCapsFilter), "caps", caps, NULL);

        m_pOsd = gst_element_factory_make(NVDS_ELEM_OSD, "osd0");
        if (!m_pOsd)
        {
            LOG_ERROR("Failed to create new OSD element for OSD '" << osd << "'");
            throw;  
        }

        g_object_set(G_OBJECT(m_pOsd), "display-clock", m_isClockEnabled,
            "clock-font", m_font, "x-clock-offset", m_clockOffsetX,
            "y-clock-offset", m_clockOffsetY, "clock-color", m_clkColor,
            "clock-font-size", m_clockFontOffset, "process-mode", m_processMode, NULL);

        gst_bin_add_many(GST_BIN(m_pBin), m_pQueue, m_pVidConv, m_pConvQueue, m_pOsd, NULL);

        g_object_set(G_OBJECT (bin->nvvidconv), "gpu-id", config->gpu_id, NULL);
        g_object_set(G_OBJECT (bin->nvvidconv), "nvbuf-memory-type",
            config->nvbuf_memory_type, NULL);

        g_object_set(G_OBJECT(m_pOsd), "gpu-id", m_gpuId, NULL);

        if (!gst_element_link(m_pQueue, m_pVidConv))
        {
            LOG_ERROR("Failed to link Queue to Vid Conv for OSD '" << osd <<" '");
            throw;
        }
        
        if (!gst_element_link(m_pVidConv, m_pConvQueue))
        {
            LOG_ERROR("Failed to link Vid Conv to Conv Queue for OSD '" << osd <<" '");
            throw;
        }
        if (!gst_element_link(m_pConvQueue, m_pOsd))
        {
            LOG_ERROR("Failed to link Conv Queue to OSD for '" << osd <<" '");
            throw;
        }

        gst_element_add_pad(m_pBin, gst_ghost_pad_new("sink", m_pQueue));
        gst_element_add_pad(m_pBin, gst_ghost_pad_new("src", m_pOsd));

    }    
    
    OsdBintr::~OsdBintr()
    {
        LOG_FUNC();
    }
}    