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

    SinkBintr::SinkBintr(const char* sink)
        : Bintr(sink)
        , m_isOverlay(false)
    {
        LOG_FUNC();
        
//        m_pStaticSinkPadtr = DSL_STATIC_PADTR_NEW("src");
    }

    SinkBintr::~SinkBintr()
    {
        LOG_FUNC();
    }

    bool SinkBintr::AddToParent(DSL_NODETR_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' Sink to the Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            AddSinkBintr(shared_from_this());
    }


    bool SinkBintr::IsParent(DSL_NODETR_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // check if 'this' Sink is child of Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            IsSinkBintrChild(std::dynamic_pointer_cast<SinkBintr>(shared_from_this()));
    }

    bool SinkBintr::RemoveFromParent(DSL_NODETR_PTR pParentBintr)
    {
        LOG_FUNC();
        
        if (!IsParent(pParentBintr))
        {
            LOG_ERROR("Sink '" << m_name << "' is not a child of Pipeline '" << pParentBintr->m_name << "'");
            return false;
        }
        // remove 'this' Sink from the Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            RemoveSinkBintr(std::dynamic_pointer_cast<SinkBintr>(shared_from_this()));
    }
    
    bool SinkBintr::IsOverlay()
    {
        LOG_FUNC();
        
        return m_isOverlay;
    }
    
    OverlaySinkBintr::OverlaySinkBintr(const char* sink, guint offsetX, guint offsetY, 
        guint width, guint height)
        : SinkBintr(sink)
        , m_sync(FALSE)
        , m_async(FALSE)
        , m_qos(TRUE)
        , m_displayId(0)
        , m_offsetX(offsetX)
        , m_offsetY(offsetY)
        , m_width(width)
        , m_height(height)
    {
        LOG_FUNC();
        
        m_isOverlay = true;

        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "sink-bin-queue");
        m_pOverlay = DSL_ELEMENT_NEW(NVDS_ELEM_SINK_OVERLAY, "sink-bin-overlay");
        
        m_pOverlay->SetAttribute("display-id", m_displayId);
        m_pOverlay->SetAttribute("overlay", false);
        m_pOverlay->SetAttribute("overlay-x", m_offsetX);
        m_pOverlay->SetAttribute("overlay-y", m_offsetY);
        m_pOverlay->SetAttribute("overlay-w", m_width);
        m_pOverlay->SetAttribute("overlay-h", m_height);
        m_pOverlay->SetAttribute("sync", m_sync);
        m_pOverlay->SetAttribute("max-lateness", -1);
        m_pOverlay->SetAttribute("async", m_async);
        m_pOverlay->SetAttribute("qos", m_qos);
        
        AddChild(m_pQueue);
        AddChild(m_pOverlay);
            
        m_pQueue->AddGhostPadToParent("sink");
    }
    
    OverlaySinkBintr::~OverlaySinkBintr()
    {
        LOG_FUNC();
    
        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }

    bool OverlaySinkBintr::LinkAll()
    {
        LOG_FUNC();
        
        m_pQueue->LinkToSink(m_pOverlay);
        
        return true;
    }
    
    void OverlaySinkBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        m_pQueue->UnlinkFromSink();
    }
    
    void OverlaySinkBintr::SetDisplayId(int id)
    {
        LOG_FUNC();
        
        m_displayId = id;
        m_pOverlay->SetAttribute("display-id", m_displayId);
    }
    
}    