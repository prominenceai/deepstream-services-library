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
#include "DsdStreamMuxBintr.h"


namespace DSD
{
    StreamMuxBintr::StreamMuxBintr(const std::string& streammux, 
        gboolean live, guint batchSize, guint batchTimeout, guint width, guint height)
        : Bintr(streammux)
        , m_width(width)
        , m_height(height)
        , m_batchSize(batchSize)
        , m_batchTimeout(batchTimeout)
        , m_isLive(live)
        , m_gpuId(0)
        , m_enablePadding(FALSE)
        , m_nvbufMemoryType(0)
    {
        LOG_FUNC();
        
        m_pBin = gst_element_factory_make(NVDS_ELEM_STREAM_MUX, "stream_muxer");
        if (!m_pBin) 
        {            
            LOG_ERROR("Failed to create new Stream Muxer bin for '" << streammux << "'");
            throw;
        };
        g_object_set(G_OBJECT(m_pBin), "gpu-id", m_gpuId, NULL);
        g_object_set(G_OBJECT(m_pBin), "nvbuf-memory-type", m_nvbufMemoryType, NULL);
        g_object_set(G_OBJECT(m_pBin), "live-source", m_isLive, NULL);
        g_object_set(G_OBJECT(m_pBin), "batched-push-timeout", m_batchTimeout, NULL);

        if ((gboolean)m_batchSize)
        {
            g_object_set(G_OBJECT(m_pBin), "batch-size", m_batchSize, NULL);
        }

        g_object_set(G_OBJECT(m_pBin), "enable-padding", m_enablePadding, NULL);

        if (m_width && m_height)
        {
            g_object_set(G_OBJECT(m_pBin), "width", m_width, NULL);
            g_object_set(G_OBJECT(m_pBin), "height", m_height, NULL);
        }
    };    

    StreamMuxBintr::~StreamMuxBintr()
    {
        LOG_FUNC();
    };
    
    
};