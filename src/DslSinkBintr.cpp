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
#include "DslSinkBintr.h"

namespace DSL
{
    SinkBintr::SinkBintr(const std::string& sink, guint displayId, guint overlayId,
        guint offsetX, guint offsetY, guint width, guint height)
        : Bintr(sink)
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

        // New Queie. Tee, amd Sink Elements for this Sink bin
        // Note!, elements will be linked in the order they're created
        
        m_pQueue = MakeElement(NVDS_ELEM_QUEUE, "sink_bin_queue", LINK_TRUE);
        m_pTee = MakeElement(NVDS_ELEM_TEE, "sink_bin_tee", LINK_TRUE);
        m_pSink = MakeElement(NVDS_ELEM_SINK_OVERLAY, (gchar*)sink.c_str(), LINK_TRUE);
        
        g_object_set(G_OBJECT(m_pSink), "display-id", m_displayId, NULL);
        g_object_set(G_OBJECT(m_pSink), "overlay", m_overlayId, NULL);
        g_object_set(G_OBJECT(m_pSink), "overlay-x", m_offsetX, NULL);
        g_object_set(G_OBJECT(m_pSink), "overlay-y", m_offsetY, NULL);
        g_object_set(G_OBJECT(m_pSink), "overlay-w", m_width, NULL);
        g_object_set(G_OBJECT(m_pSink), "overlay-h", m_height, NULL);

        AddGhostPads();
    }
    
    SinkBintr::~SinkBintr()
    {
        LOG_FUNC();
    }
}    