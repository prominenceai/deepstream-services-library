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
#include "DslSourceBintr.h"

#define N_DECODE_SURFACES 16
#define N_EXTRA_SURFACES 1

namespace DSL
{
    SourceBintr::SourceBintr(const std::string& source, guint type, gboolean live, 
        guint width, guint height, guint fps_n, guint fps_d)
        : Bintr(source)
        , m_type(type)
        , m_isLive(live)
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
        m_pCapsFilter = MakeElement(NVDS_ELEM_CAPS_FILTER, "src_cap_filter");

        GstCaps * pCaps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12",
            "width", G_TYPE_INT, m_width, "height", G_TYPE_INT, m_height, 
            "framerate", GST_TYPE_FRACTION, m_fps_n, m_fps_d, NULL);
        if (!pCaps)
        {
            LOG_ERROR("Failed to create new Caps Simple for '" << source << "'");
            throw;  
        }

        g_object_set(G_OBJECT(m_pSourceElement), "bufapi-version", TRUE, NULL);
        g_object_set(G_OBJECT(m_pSourceElement), "maxperf", TRUE, NULL);
        g_object_set(G_OBJECT(m_pSourceElement), "sensor-id", 1, NULL);

        GstCapsFeatures *feature = NULL;
        feature = gst_caps_features_new("memory:NVMM", NULL);

        gst_caps_set_features(pCaps, 0, feature);
        g_object_set(G_OBJECT(pCapsFilter), "caps", pCaps, NULL);
        
        gst_caps_unref(pCaps);        
        
        AddGhostPads();
    }

    SourceBintr::~SourceBintr()
    {
        LOG_FUNC();

    }
}