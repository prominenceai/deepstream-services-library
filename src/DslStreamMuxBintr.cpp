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
#include "DslStreamMuxBintr.h"


namespace DSL
{
    StreamMuxBintr::StreamMuxBintr(const std::string& streammux, 
        gboolean live, guint batchSize, guint batchTimeout, guint width, guint height)
        : Bintr(streammux)
        , m_width(width)
        , m_height(height)
        , m_batchSize(batchSize)
        , m_batchTimeout(batchTimeout)
        , m_isLive(live)
        , m_enablePadding(FALSE)
    {
        LOG_FUNC();
        
        m_pStreamMux = MakeElement(NVDS_ELEM_STREAM_MUX, "stream_muxer", LINK_FALSE);
        
        g_object_set(G_OBJECT(m_pStreamMux), "gpu-id", m_gpuId, NULL);
        g_object_set(G_OBJECT(m_pStreamMux), "nvbuf-memory-type", m_nvbufMemoryType, NULL);
        g_object_set(G_OBJECT(m_pStreamMux), "live-source", m_isLive, NULL);
        g_object_set(G_OBJECT(m_pStreamMux), "batched-push-timeout", m_batchTimeout, NULL);

        if ((gboolean)m_batchSize)
        {
            g_object_set(G_OBJECT(m_pStreamMux), "batch-size", m_batchSize, NULL);
        }

        g_object_set(G_OBJECT(m_pStreamMux), "enable-padding", m_enablePadding, NULL);

        if (m_width && m_height)
        {
            g_object_set(G_OBJECT(m_pStreamMux), "width", m_width, NULL);
            g_object_set(G_OBJECT(m_pStreamMux), "height", m_height, NULL);
        }
    };    

    StreamMuxBintr::~StreamMuxBintr()
    {
        LOG_FUNC();
    };
    
    void StreamMuxBintr::AddToParent(Bintr* pParentBintr)
    {
        LOG_FUNC();

        ((PipelineBintr*)pParentBintr)->AddStreamMuxBintr(this*);
    }
    
};