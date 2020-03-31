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
#include "DslElementr.h"
#include "DslOsdBintr.h"
#include "DslPipelineBintr.h"

namespace DSL
{

    OsdBintr::OsdBintr(const char* name, gboolean isClockEnabled)
        : Bintr(name)
        , m_isClockEnabled(isClockEnabled)
        , m_processMode(0)
        , m_cropLeft(0)
        , m_cropTop(0)
        , m_cropWidth(0)
        , m_cropHeight(0)
        , m_clockFont("Serif")
        , m_clockFontSize(12)
        , m_clockOffsetX(0)
        , m_clockOffsetY(0)
        , m_clockColorRed(0)
        , m_clockColorGreen(0)
        , m_clockColorBlue(0)
        , m_streamId(-1)
    {
        LOG_FUNC();
        
        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "osd_queue");
        m_pVidPreConv = DSL_ELEMENT_NEW(NVDS_ELEM_VIDEO_CONV, "osd_vid_pre_conv");
//        m_pConvQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "osd_conv_queue");
        m_pOsd = DSL_ELEMENT_NEW(NVDS_ELEM_OSD, "nvosd0");

        m_pVidPreConv->SetAttribute("gpu-id", m_gpuId);
        m_pVidPreConv->SetAttribute("nvbuf-memory-type", m_nvbufMemoryType);

        m_pOsd->SetAttribute("gpu-id", m_gpuId);
        m_pOsd->SetAttribute("display-clock", m_isClockEnabled);
        m_pOsd->SetAttribute("clock-font", m_clockFont.c_str()); 
        m_pOsd->SetAttribute("x-clock-offset", m_clockOffsetX);
        m_pOsd->SetAttribute("y-clock-offset", m_clockOffsetY);
        m_pOsd->SetAttribute("clock-font-size", m_clockFontSize);
        m_pOsd->SetAttribute("process-mode", m_processMode);
        
//        SetClockColor(m_clockColorRed, m_clockColorGreen, m_clockColorBlue);
        
        AddChild(m_pQueue);
        AddChild(m_pVidPreConv);
//        AddChild(m_pConvQueue);
        AddChild(m_pOsd);

        m_pQueue->AddGhostPadToParent("sink");
        m_pOsd->AddGhostPadToParent("src");

        m_pSinkPadProbe = DSL_PAD_PROBE_NEW("osd-sink-pad-probe", "sink", m_pQueue);
        m_pSrcPadProbe = DSL_PAD_PROBE_NEW("osd-src-pad-probe", "src", m_pOsd);
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
//            !m_pVidPreConv->LinkToSink(m_pConvQueue) or
//            !m_pConvQueue->LinkToSink(m_pOsd))
            !m_pVidPreConv->LinkToSink(m_pOsd))
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
//        m_pConvQueue->UnlinkFromSink();
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

    bool OsdBintr::AddToParent(DSL_NODETR_PTR pBranchBintr)
    {
        LOG_FUNC();
        
        // add 'this' OSD to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pBranchBintr)->
            AddOsdBintr(shared_from_this());
    }

    void OsdBintr::GetClockEnabled(boolean* enabled)
    {
        LOG_FUNC();
        
        *enabled = m_isClockEnabled;
    }
    
    bool OsdBintr::SetClockEnabled(boolean enabled)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to Set the Clock Enabled attribute for OsdBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }
        m_isClockEnabled = enabled;
        m_pOsd->SetAttribute("display-clock", m_isClockEnabled);
        
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
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Clock Offsets for OsdBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

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
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Clock Font for OsdBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_clockFont.assign(name);
        m_clockFontSize = size;
        
        m_pOsd->SetAttribute("clock-font", m_clockFont.c_str()); 
        m_pOsd->SetAttribute("clock-font-size", m_clockFontSize);
        
        return true;
    }

    void OsdBintr::GetClockColor(uint* red, uint* green, uint* blue)
    {
        LOG_FUNC();
        
        *red = m_clockColorRed;
        *green = m_clockColorGreen;
        *blue = m_clockColorBlue;
    }

    bool OsdBintr::SetClockColor(uint red, uint green, uint blue)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Clock Color for OsdBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        // TODO how best to handle colors... follow NVIDIA and use floats or
        // stick with all ints for params and scale ?????
        // m_pOsd->SetAttribute("clock-color", clockColor);
        
        return true;
    }

    bool OsdBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set GPU ID for OsdBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_gpuId = gpuId;
        LOG_DEBUG("Setting GPU ID to '" << gpuId << "' for OsdBintr '" << m_name << "'");

        m_pVidPreConv->SetAttribute("gpu-id", m_gpuId);
        m_pOsd->SetAttribute("gpu-id", m_gpuId);
        
        return true;
    }

    void OsdBintr::GetCropSettings(uint *left, uint *top, uint *width, uint *height)
    {
        LOG_FUNC();
        
        *left = m_cropLeft;
        *top = m_cropTop;
        *width = m_cropWidth;
        *height = m_cropHeight;
    }

    static GstPadProbeReturn IdlePadProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pOsdBintr)
    {
        static_cast<OsdBintr*>(pOsdBintr)->UpdateCropSetting();
        return GST_PAD_PROBE_REMOVE;
    }
    
    bool OsdBintr::SetCropSettings(uint left, uint top, uint width, uint height)
    {
        LOG_FUNC();
        
        m_cropLeft = left;
        m_cropTop = top;
        m_cropWidth = width;
        m_cropHeight = height;
        
        if (IsLinked())
        {
            GstPad* pStaticPad = gst_element_get_static_pad(m_pVidPreConv->GetGstElement(), "src");
            if (!pStaticPad)
            {
                LOG_ERROR("Failed to get Static Pad for OsdBintr '" << GetName() << "'");
                return false;
            }
            gst_pad_add_probe(pStaticPad, GST_PAD_PROBE_TYPE_IDLE, IdlePadProbeCB, this, NULL);
            gst_object_unref(pStaticPad);
            
            return true;
        }

        std::string pixelSet = std::to_string(m_cropLeft) + ":" + std::to_string(m_cropTop) + 
            ":" + std::to_string(m_cropWidth)  + ":" + std::to_string(m_cropHeight);
            
        m_pVidPreConv->SetAttribute("src-crop", pixelSet.c_str());
        m_pVidPreConv->SetAttribute("dest-crop", pixelSet.c_str());

        return true;
    }

    static GstPadProbeReturn ConsumeEosCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pOsdBintr)
    {
        if (GST_EVENT_TYPE(GST_PAD_PROBE_INFO_DATA(pInfo)) != GST_EVENT_EOS)
        {
            return GST_PAD_PROBE_PASS;
        }
        // remove the probe first
        gst_pad_remove_probe(pPad, GST_PAD_PROBE_INFO_ID(pInfo));
        return GST_PAD_PROBE_DROP;
    }

    void OsdBintr::UpdateCropSetting()
    {
        GstPad* pConvStaticSrcPad = gst_element_get_static_pad(m_pVidPreConv->GetGstElement(), "src");
        GstPad* pOsdStaticSinkPad = gst_element_get_static_pad(m_pOsd->GetGstElement(), "sink");
        
        gst_pad_unlink(pConvStaticSrcPad, pOsdStaticSinkPad);
            
        std::string pixelSet = std::to_string(m_cropLeft) + ":" + std::to_string(m_cropTop) + 
            ":" + std::to_string(m_cropWidth)  + ":" + std::to_string(m_cropHeight);
            
        m_pVidPreConv->SetAttribute("src-crop", pixelSet.c_str());
        m_pVidPreConv->SetAttribute("dest-crop", pixelSet.c_str());

        gst_pad_add_probe(pOsdStaticSinkPad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, ConsumeEosCB, this, NULL);

        gst_pad_link(pConvStaticSrcPad, pOsdStaticSinkPad);

        gst_object_unref(pConvStaticSrcPad);
        gst_object_unref(pOsdStaticSinkPad);
    }    
}    