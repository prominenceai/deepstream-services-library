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
        , m_isWindowCapable(false)
        , m_sinkId(-1)
    {
        LOG_FUNC();

        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "sink-bin-queue");
        AddChild(m_pQueue);
        m_pQueue->AddGhostPadToParent("sink");
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
            LOG_ERROR("Sink '" << GetName() << "' is not a child of Pipeline '" << pParentBintr->GetName() << "'");
            return false;
        }
        // remove 'this' Sink from the Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            RemoveSinkBintr(std::dynamic_pointer_cast<SinkBintr>(shared_from_this()));
    }

    int SinkBintr::GetSinkId()
    {
        LOG_FUNC();
        
        return m_sinkId;
    }

    void SinkBintr::SetSinkId(int id)
    {
        LOG_FUNC();

        m_sinkId = id;
    }
    
    bool SinkBintr::IsWindowCapable()
    {
        LOG_FUNC();
        
        return m_isWindowCapable;
    }
    
    bool SinkBintr::LinkToSource(DSL_NODETR_PTR pTee)
    {
        LOG_FUNC();

        std::string srcPadName = "src_" + std::to_string(m_sinkId);

        LOG_INFO("Linking Sink '" << GetName() << "' to Pad '" << srcPadName
            << "' for Tee '" << pTee->GetName() << "'");
        
        m_pGstStaticSinkPad = gst_element_get_static_pad(GetGstElement(), "sink");
        if (!m_pGstStaticSinkPad)
        {
            LOG_ERROR("Failed to get Static Sink Pad for SinkBintr '" << GetName() << "'");
            return false;
        }

        GstPad* pGstRequestedSourcePad = gst_element_get_request_pad(pTee->GetGstElement(), "src_%u");
            
        if (!pGstRequestedSourcePad)
        {
            LOG_ERROR("Failed to get Tee Pad for PipelineSinksBintr '" << GetName() <<" '");
            return false;
        }

        m_pGstRequestedSourcePads[srcPadName] = pGstRequestedSourcePad;

        return Bintr::LinkToSource(pTee);
        
    }
    
    bool SinkBintr::UnlinkFromSource()
    {
        LOG_FUNC();
        
        // If we're not currently linked to the Tee
        if (!IsLinkedToSource())
        {
            LOG_ERROR("SinkBintr '" << GetName() << "' is not in a Linked state");
            return false;
        }

        std::string srcPadName = "src_" + std::to_string(m_sinkId);

        LOG_INFO("Unlinking and releasing requested Source Pad for Sink Tee " << GetName());
        
        gst_pad_unlink(m_pGstRequestedSourcePads[srcPadName], m_pGstStaticSinkPad);
        gst_element_release_request_pad(GetSource()->GetGstElement(), m_pGstRequestedSourcePads[srcPadName]);
                
        m_pGstRequestedSourcePads.erase(srcPadName);
        
        return Nodetr::UnlinkFromSource();
    }
    
    OverlaySinkBintr::OverlaySinkBintr(const char* sink, guint offsetX, guint offsetY, 
        guint width, guint height)
        : SinkBintr(sink)
        , m_sync(TRUE)
        , m_async(FALSE)
        , m_qos(TRUE)
        , m_displayId(0)
        , m_overlayId(0)
        , m_offsetX(offsetX)
        , m_offsetY(offsetY)
        , m_width(width)
        , m_height(height)
    {
        LOG_FUNC();
        
        m_isWindowCapable = false;

        m_pOverlay = DSL_ELEMENT_NEW(NVDS_ELEM_SINK_OVERLAY, "sink-bin-overlay");
//        m_pOverlay->SetAttribute("overlay", m_overlayId);
        m_pOverlay->SetAttribute("overlay-x", m_offsetX);
        m_pOverlay->SetAttribute("overlay-y", m_offsetY);
        m_pOverlay->SetAttribute("overlay-w", m_width);
        m_pOverlay->SetAttribute("overlay-h", m_height);
        m_pOverlay->SetAttribute("sync", m_sync);
        m_pOverlay->SetAttribute("max-lateness", -1);
        m_pOverlay->SetAttribute("async", m_async);
        m_pOverlay->SetAttribute("qos", m_qos);
        
        AddChild(m_pOverlay);
    }
    
    OverlaySinkBintr::~OverlaySinkBintr()
    {
        LOG_FUNC();
    
        if (IsLinked())
        {    
            UnlinkAll();
        }
    }

    bool OverlaySinkBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("OverlaySinkBintr '" << m_name << "' is already linked");
            return false;
        }
        if (!m_pQueue->LinkToSink(m_pOverlay))
        {
            return false;
        }
        m_isLinked = true;
        return true;
    }
    
    void OverlaySinkBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("OverlaySinkBintr '" << m_name << "' is not linked");
            return;
        }
        m_pQueue->UnlinkFromSink();
        m_isLinked = false;
    }

    int OverlaySinkBintr::GetDisplayId()
    {
        LOG_FUNC();
        
        return m_displayId;
    }
    
    void OverlaySinkBintr::SetDisplayId(int id)
    {
        LOG_FUNC();
        
        m_displayId = id;
        m_pOverlay->SetAttribute("display-id", m_displayId);
    }
    
    void  OverlaySinkBintr::GetOffsets(uint* offsetX, uint* offsetY)
    {
        LOG_FUNC();
        
        *offsetX = m_offsetX;
        *offsetY = m_offsetY;
    }
    
    bool OverlaySinkBintr::SetOffsets(uint offsetX, uint offsetY)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Dimensions for OverlaySinkBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_offsetX = offsetX;
        m_offsetY = offsetY;

        m_pOverlay->SetAttribute("overlay-x", m_offsetX);
        m_pOverlay->SetAttribute("overlay-y", m_offsetY);
        
        return true;
    }

    void OverlaySinkBintr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_width;
        *height = m_height;
    }

    bool OverlaySinkBintr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Dimensions for OverlaySinkBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_width = width;
        m_height = height;

        m_pOverlay->SetAttribute("overlay-w", m_width);
        m_pOverlay->SetAttribute("overlay-h", m_height);
        
        return true;
    }
    
    WindowSinkBintr::WindowSinkBintr(const char* sink, guint offsetX, guint offsetY, 
        guint width, guint height)
        : SinkBintr(sink)
        , m_sync(TRUE)
        , m_async(FALSE)
        , m_qos(TRUE)
        , m_offsetX(offsetX)
        , m_offsetY(offsetY)
        , m_width(width)
        , m_height(height)
        , m_pXWindow(0)
    {
        LOG_FUNC();
        
        m_isWindowCapable = true;

        m_pTransform = DSL_ELEMENT_NEW(NVDS_ELEM_EGLTRANSFORM, "sink-bin-transform");
        m_pEglGles = DSL_ELEMENT_NEW(NVDS_ELEM_SINK_EGL, "sink-bin-eglgles");
        
        m_pEglGles->SetAttribute("window-x", m_offsetX);
        m_pEglGles->SetAttribute("window-y", m_offsetY);
        m_pEglGles->SetAttribute("window-width", m_width);
        m_pEglGles->SetAttribute("window-height", m_height);
        m_pEglGles->SetAttribute("sync", m_sync);
        m_pEglGles->SetAttribute("max-lateness", -1);
        m_pEglGles->SetAttribute("async", m_async);
        m_pEglGles->SetAttribute("qos", m_qos);
        
        AddChild(m_pTransform);
        AddChild(m_pEglGles);
    }
    
    WindowSinkBintr::~WindowSinkBintr()
    {
        LOG_FUNC();
    
        if (IsLinked())
        {    
            UnlinkAll();
        }
    }

    bool WindowSinkBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("OverlaySinkBintr '" << m_name << "' is already linked");
            return false;
        }
        if (!m_pQueue->LinkToSink(m_pTransform) or
            !m_pTransform->LinkToSink(m_pEglGles))
        {
            return false;
        }
        m_isLinked = true;
        return true;
    }
    
    void WindowSinkBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("OverlaySinkBintr '" << m_name << "' is not linked");
            return;
        }
        m_pQueue->UnlinkFromSink();
        m_pTransform->UnlinkFromSink();
        m_isLinked = false;
    }

    void  WindowSinkBintr::GetOffsets(uint* offsetX, uint* offsetY)
    {
        LOG_FUNC();
        
        *offsetX = m_offsetX;
        *offsetY = m_offsetY;
    }
    
    bool WindowSinkBintr::SetOffsets(uint offsetX, uint offsetY)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Dimensions for WindowSinkBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_offsetX = offsetX;
        m_offsetY = offsetY;

        m_pEglGles->SetAttribute("window-x", m_offsetX);
        m_pEglGles->SetAttribute("window-y", m_offsetY);
        
        return true;
    }

    void WindowSinkBintr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_width;
        *height = m_height;
    }

    bool WindowSinkBintr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Dimensions for WindowSinkBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_width = width;
        m_height = height;

        m_pEglGles->SetAttribute("window-width", m_width);
        m_pEglGles->SetAttribute("window-height", m_height);
        
        return true;
    }

    Window WindowSinkBintr::GetXWindowHandle()
    {
        LOG_FUNC();
        
        return m_pXWindow;
    }
    
    void WindowSinkBintr::SetXWindowHandle(Window pXWindow)
    {
        LOG_FUNC();
        
        m_pXWindow = pXWindow;
        
        LOG_WARN("Window handle = " << m_pXWindow);
        gst_video_overlay_set_window_handle(
            GST_VIDEO_OVERLAY(m_pEglGles->GetGstElement()), m_pXWindow);
        gst_video_overlay_expose(GST_VIDEO_OVERLAY(m_pEglGles->GetGstElement()));
    }

}    