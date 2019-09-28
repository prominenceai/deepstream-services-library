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
#include "DslOsdBintr.h"
#include "DslPipelineBintr.h"

namespace DSL
{
    std::string OsdBintr::m_sClockFont = "Serif";
    guint OsdBintr::m_sClockFontSize = 12;
    guint OsdBintr::m_sClockOffsetX = 800;
    guint OsdBintr::m_sClockOffsetY = 820;
    guint OsdBintr::m_sClockColor = 0;
    
    OsdBintr::OsdBintr(const char* osd, gboolean isClockEnabled)
        : Bintr(osd)
        , m_isClockEnabled(isClockEnabled)
        , m_processMode(0)
        , m_pQueue(NULL)
        , m_pVidConv(NULL)
        , m_pCapsFilter(NULL)
        , m_pConvQueue(NULL)
        , m_pOsd(NULL)
    {
        LOG_FUNC();
        
        m_pQueue = MakeElement(NVDS_ELEM_QUEUE, "osd_queue", LINK_TRUE);
        m_pVidConv = MakeElement(NVDS_ELEM_VIDEO_CONV, "osd_conv", LINK_TRUE);
        m_pConvQueue = MakeElement(NVDS_ELEM_QUEUE, "osd_conv_queue", LINK_TRUE);
        m_pOsd = MakeElement(NVDS_ELEM_OSD, "nvosd0", LINK_TRUE);
        m_pCapsFilter = MakeElement(NVDS_ELEM_CAPS_FILTER, "osd_caps", LINK_FALSE);
        
        GstCaps *caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGBA", NULL);
        GstCapsFeatures *feature = gst_caps_features_new(MEMORY_FEATURES, NULL);

        gst_caps_set_features(caps, 0, feature);
        g_object_set(G_OBJECT(m_pCapsFilter), "caps", caps, NULL);

        g_object_set(G_OBJECT(m_pOsd), "display-clock", m_isClockEnabled,
            "clock-font", (gchar*)m_sClockFont.c_str(), "x-clock-offset", m_sClockOffsetX,
            "y-clock-offset", m_sClockOffsetY, "clock-color", m_sClockColor,
            "clock-font-size", m_sClockFontSize, "process-mode", m_processMode, NULL);

        g_object_set(G_OBJECT(m_pVidConv), "gpu-id", m_gpuId, NULL);
        g_object_set(G_OBJECT(m_pVidConv), "nvbuf-memory-type", m_nvbufMemoryType, NULL);

        g_object_set(G_OBJECT(m_pOsd), "gpu-id", m_gpuId, NULL);

        AddGhostPads();
    }    
    
    OsdBintr::~OsdBintr()
    {
        LOG_FUNC();
    }

    void OsdBintr::AddToParent(std::shared_ptr<Bintr> pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' OSD to the Parent Pipeline 
        std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)-> \
            AddOsdBintr(shared_from_this());
    }
}    