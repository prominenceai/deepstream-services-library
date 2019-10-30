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
#include "DslSinkBintr.h"
#include "DslPipelineBintr.h"

namespace DSL
{

    SinksBintr::SinksBintr(const char* sink)
        : Bintr(sink)
        , m_pQueue(NULL)
        , m_pTee(NULL)
    {
        LOG_FUNC();

        m_pQueue = std::shared_ptr<Elementr>(new Elementr(NVDS_ELEM_QUEUE, "sink_bin_queue", m_pBin));
        m_pTee = std::shared_ptr<Elementr>(new Elementr(NVDS_ELEM_TEE, "sink_bin_tee", m_pBin));
        
        m_pQueue->AddSinkGhostPad();
    }
    
    SinksBintr::~SinksBintr()
    {
        LOG_FUNC();
    }
     
    void SinksBintr::AddChild(std::shared_ptr<Bintr> pChildBintr)
    {
        LOG_FUNC();
        
        pChildBintr->m_pParentBintr = 
            std::dynamic_pointer_cast<Bintr>(shared_from_this());

        m_pChildBintrs[pChildBintr->m_name] = pChildBintr;
                        
        if (!gst_bin_add(GST_BIN(m_pBin), pChildBintr->m_pBin))
        {
            LOG_ERROR("Failed to add " << pChildBintr->m_name << " to " << m_name);
            throw;
        }

        GstPadTemplate* padtemplate = 
            gst_element_class_get_pad_template(GST_ELEMENT_GET_CLASS(m_pTee->m_pElement), "src_%u");
        if (!padtemplate)
        {
            LOG_ERROR("Failed to get Pad Template for '" << m_name << "'");
            throw;
        }
        
        std::shared_ptr<RequestPadtr> pSourcePadtr = 
            std::shared_ptr<RequestPadtr>(new RequestPadtr(m_pTee->m_pElement, 
            padtemplate, "src")); // Name is for Padr only, Pad name is derived from the Pad Template
        
        pSourcePadtr->LinkTo(std::dynamic_pointer_cast<SinkBintr>(pChildBintr)->m_pStaticSinkPadtr);
    };

    OverlaySinkBintr::OverlaySinkBintr(const char* sink, guint offsetX, guint offsetY, 
        guint width, guint height)
        : SinkBintr(sink)
        , m_sync(FALSE)
        , m_async(FALSE)
        , m_qos(TRUE)
        , m_displayId(-1)
        , m_overlayId(0)
        , m_offsetX(offsetX)
        , m_offsetY(offsetY)
        , m_width(width)
        , m_height(height)
    {
        LOG_FUNC();

        m_pQueue = std::shared_ptr<Elementr>(new Elementr(NVDS_ELEM_QUEUE, "sink-bin-queue", m_pBin));
        m_pOverlay = std::shared_ptr<Elementr>(new Elementr(NVDS_ELEM_SINK_OVERLAY, "sink-bin-overlay", m_pBin));
        
        g_object_set(G_OBJECT(m_pOverlay->m_pElement), 
            "overlay-x", m_offsetX,
            "overlay-y", m_offsetY,
            "overlay-w", m_width,
            "overlay-h", m_height,
            "sync", m_sync, 
            "max-lateness", -1,
            "async", m_async, 
            "qos", m_qos, NULL);
            
        m_pQueue->AddSinkGhostPad();
    
        m_pStaticSinkPadtr = std::shared_ptr<StaticPadtr>(new StaticPadtr(m_pBin, "sink"));
    }
    
    OverlaySinkBintr::~OverlaySinkBintr()
    {
        LOG_FUNC();
    }

    void OverlaySinkBintr::LinkAll()
    {
        LOG_FUNC();
        
        m_pQueue->LinkTo(m_pOverlay);
    }
    
    void OverlaySinkBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        m_pQueue->Unlink();
    }

    
    void OverlaySinkBintr::AddToParent(std::shared_ptr<Bintr> pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' Sink to the Parent Pipeline 
        std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            AddSinkBintr(shared_from_this());
    }
        
}    