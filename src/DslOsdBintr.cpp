/*
The MIT License

Copyright (c) 2019-2021, Prominence AI, Inc.

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
#include "DslElementr.h"
#include "DslOsdBintr.h"
#include "DslPipelineBintr.h"

namespace DSL
{

    OsdBintr::OsdBintr(const char* name, boolean textEnabled, boolean clockEnabled)
        : Bintr(name)
        , m_textEnabled(textEnabled)
        , m_clockEnabled(clockEnabled)
        , m_processMode(0)
        , m_clockFont("Serif")
        , m_clockFontSize(12)
        , m_clockOffsetX(0)
        , m_clockOffsetY(0)
        , m_clockColor({})
        , m_streamId(-1)
    {
        LOG_FUNC();
        
        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "osd_queue");
        m_pVidPreConv = DSL_ELEMENT_NEW(NVDS_ELEM_VIDEO_CONV, "osd_vid_pre_conv");
        m_pConvQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "osd_conv_queue");
        m_pOsd = DSL_ELEMENT_NEW(NVDS_ELEM_OSD, "nvosd0");

        m_pVidPreConv->SetAttribute("gpu-id", m_gpuId);
        m_pVidPreConv->SetAttribute("nvbuf-memory-type", m_nvbufMemType);

        m_pOsd->SetAttribute("gpu-id", m_gpuId);
        m_pOsd->SetAttribute("display-text", m_textEnabled);
        m_pOsd->SetAttribute("display-clock", m_clockEnabled);
        m_pOsd->SetAttribute("clock-font", m_clockFont.c_str()); 
        m_pOsd->SetAttribute("x-clock-offset", m_clockOffsetX);
        m_pOsd->SetAttribute("y-clock-offset", m_clockOffsetY);
        m_pOsd->SetAttribute("clock-font-size", m_clockFontSize);
        m_pOsd->SetAttribute("process-mode", m_processMode);
        
        AddChild(m_pQueue);
        AddChild(m_pVidPreConv);
        AddChild(m_pConvQueue);
        AddChild(m_pOsd);

        m_pQueue->AddGhostPadToParent("sink");
        m_pOsd->AddGhostPadToParent("src");

        m_pSinkPadProbe = DSL_PAD_BUFFER_PROBE_NEW("osd-sink-pad-probe", "sink", m_pQueue);
        m_pSrcPadProbe = DSL_PAD_BUFFER_PROBE_NEW("osd-src-pad-probe", "src", m_pOsd);
    }    
    
    OsdBintr::~OsdBintr()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }

    bool OsdBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("OsdBintr '" << m_name << "' is already linked");
            return false;
        }
        
        if (!m_pQueue->LinkToSink(m_pVidPreConv) or
            !m_pVidPreConv->LinkToSink(m_pConvQueue) or
            !m_pConvQueue->LinkToSink(m_pOsd))
        {
            return false;
        }
        m_isLinked = true;
        return true;
    }
    
    void OsdBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("OsdBintr '" << m_name << "' is not linked");
            return;
        }
        m_pQueue->UnlinkFromSink();
        m_pVidPreConv->UnlinkFromSink();
        m_pConvQueue->UnlinkFromSink();
        m_isLinked = false;
    }

    bool OsdBintr::LinkToSource(DSL_NODETR_PTR pDemuxer)
    {
        LOG_FUNC();
        
        std::string srcPadName = "src_" + std::to_string(m_streamId);
        
        LOG_INFO("Linking the OsdBintr '" << GetName() << "' to Pad '" << srcPadName 
            << "' for Demuxer '" << pDemuxer->GetName() << "'");
       
        m_pGstStaticSinkPad = gst_element_get_static_pad(GetGstElement(), "sink");
        if (!m_pGstStaticSinkPad)
        {
            LOG_ERROR("Failed to get Static Sink Pad for OsdBintr '" << GetName() << "'");
            return false;
        }

        GstPad* pGstRequestedSrcPad = gst_element_get_request_pad(pDemuxer->GetGstElement(), srcPadName.c_str());
            
        if (!pGstRequestedSrcPad)
        {
            LOG_ERROR("Failed to get Requested Src Pad for Demuxer '" << pDemuxer->GetName() << "'");
            return false;
        }
        m_pGstRequestedSourcePads[srcPadName] = pGstRequestedSrcPad;

        // Call the base class to complete the link relationship
        return Bintr::LinkToSource(pDemuxer);
    }
    
    bool OsdBintr::UnlinkFromSource()
    {
        LOG_FUNC();
        
        // If we're not currently linked to the Demuxer
        if (!IsLinkedToSource())
        {
            LOG_ERROR("OsdBintr '" << GetName() << "' is not in a Linked state");
            return false;
        }

        std::string srcPadName = "src_" + std::to_string(m_streamId);

        LOG_INFO("Unlinking and releasing requested Src Pad for Demuxer");
        
        gst_pad_unlink(m_pGstRequestedSourcePads[srcPadName], m_pGstStaticSinkPad);
        gst_element_release_request_pad(GetSource()->GetGstElement(), m_pGstRequestedSourcePads[srcPadName]);
                
        m_pGstRequestedSourcePads.erase(srcPadName);
        
        return Nodetr::UnlinkFromSource();
    }

    bool OsdBintr::AddToParent(DSL_BASE_PTR pBranchBintr)
    {
        LOG_FUNC();
        
        // add 'this' OSD to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pBranchBintr)->
            AddOsdBintr(shared_from_this());
    }

    bool OsdBintr::RemoveFromParent(DSL_BASE_PTR pBranchBintr)
    {
        LOG_FUNC();
        
        // remove 'this' OSD to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pBranchBintr)->
            RemoveOsdBintr(shared_from_this());
    }

    void OsdBintr::GetTextEnabled(boolean* enabled)
    {
        LOG_FUNC();
        
        *enabled = m_textEnabled;
    }
    
    bool OsdBintr::SetTextEnabled(boolean enabled)
    {
        LOG_FUNC();
        
//        if (IsInUse())
//        {
//            LOG_ERROR("Unable to Set the Text Enabled attribute for OsdBintr '" << GetName() 
//                << "' as it's currently in use");
//            return false;
//        }
        m_textEnabled = enabled;
        m_pOsd->SetAttribute("display-text", m_textEnabled);
        
        return true;
    }

    void OsdBintr::GetClockEnabled(boolean* enabled)
    {
        LOG_FUNC();
        
        *enabled = m_clockEnabled;
    }
    
    bool OsdBintr::SetClockEnabled(boolean enabled)
    {
        LOG_FUNC();
        
//        if (IsInUse())
//        {
//            LOG_ERROR("Unable to Set the Clock Enabled attribute for OsdBintr '" << GetName() 
//                << "' as it's currently in use");
//            return false;
//        }
        m_clockEnabled = enabled;
        m_pOsd->SetAttribute("display-clock", m_clockEnabled);
        
        return true;
    }

    void  OsdBintr::GetClockOffsets(uint* offsetX, uint* offsetY)
    {
        LOG_FUNC();
        
        *offsetX = m_clockOffsetX;
        *offsetY = m_clockOffsetY;
    }
    
    bool OsdBintr::SetClockOffsets(uint offsetX, uint offsetY)
    {
        LOG_FUNC();
        
//        if (IsInUse())
//        {
//            LOG_ERROR("Unable to set Clock Offsets for OsdBintr '" << GetName() 
//                << "' as it's currently in use");
//            return false;
//        }

        m_clockOffsetX = offsetX;
        m_clockOffsetY = offsetY;

        m_pOsd->SetAttribute("x-clock-offset", m_clockOffsetX);
        m_pOsd->SetAttribute("y-clock-offset", m_clockOffsetY);
        
        return true;
    }

    void OsdBintr::GetClockFont(const char** name, uint *size)
    {
        LOG_FUNC();

        *size = m_clockFontSize;
        *name = m_clockFont.c_str();
    }

    bool OsdBintr::SetClockFont(const char* name, uint size)
    {
        LOG_FUNC();
        
//        if (IsInUse())
//        {
//            LOG_ERROR("Unable to set Clock Font for OsdBintr '" << GetName() 
//                << "' as it's currently in use");
//            return false;
//        }

        m_clockFont.assign(name);
        m_clockFontSize = size;
        
        m_pOsd->SetAttribute("clock-font", m_clockFont.c_str()); 
        m_pOsd->SetAttribute("clock-font-size", m_clockFontSize);
        
        return true;
    }

    void OsdBintr::GetClockColor(double* red, double* green, double* blue, double* alpha)
    {
        LOG_FUNC();
        
        *red = m_clockColor.red;
        *green = m_clockColor.green;
        *blue = m_clockColor.blue;
        *alpha = m_clockColor.alpha;
    }

    bool OsdBintr::SetClockColor(double red, double green, double blue, double alpha)
    {
        LOG_FUNC();
        
//        if (IsInUse())
//        {
//            LOG_ERROR("Unable to set Clock Color for OsdBintr '" << GetName() 
//                << "' as it's currently in use");
//            return false;
//        }

        m_clockColor.red = red;
        m_clockColor.green = green;
        m_clockColor.blue = blue;
        m_clockColor.alpha = alpha;

        uint clkRgbaColor =
            ((((uint) (m_clockColor.red * 255)) & 0xFF) << 24) |
            ((((uint) (m_clockColor.green * 255)) & 0xFF) << 16) |
            ((((uint) (m_clockColor.blue * 255)) & 0xFF) << 8) | 
            ((((uint) (m_clockColor.alpha * 255)) & 0xFF));

        m_pOsd->SetAttribute("clock-color", clkRgbaColor);

        return true;
    }

    bool OsdBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Unable to set GPU ID for OsdBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }

        m_gpuId = gpuId;

        m_pVidPreConv->SetAttribute("gpu-id", m_gpuId);
        m_pOsd->SetAttribute("gpu-id", m_gpuId);
        
        return true;
    }

    bool OsdBintr::SetNvbufMemType(uint nvbufMemType)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Unable to set NVIDIA buffer memory type for OsdBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_nvbufMemType = nvbufMemType;
        m_pVidPreConv->SetAttribute("nvbuf-memory-type", m_nvbufMemType);

        return true;
    }
}    