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
    OsdBintr::OsdBintr(const char* osd, gboolean isClockEnabled)
        : Bintr(osd)
        , m_isClockEnabled(isClockEnabled)
        , m_processMode(0)
        , m_clockFont("Serif")
        , m_clockFontSize(12)
        , m_clockOffsetX(800)
        , m_clockOffsetY(820)
        , m_clockColorRed(0)
        , m_clockColorGreen(0)
        , m_clockColorBlue(0)
    {
        LOG_FUNC();
        
        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "osd_queue");
        m_pVidConv = DSL_ELEMENT_NEW(NVDS_ELEM_VIDEO_CONV, "osd_conv");
        m_pConvQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "osd_conv_queue");
        m_pOsd = DSL_ELEMENT_NEW(NVDS_ELEM_OSD, "nvosd0");

        m_pVidConv->SetAttribute("gpu-id", m_gpuId);
        m_pVidConv->SetAttribute("nvbuf-memory-type", m_nvbufMemoryType);

        m_pOsd->SetAttribute("gpu-id", m_gpuId);
        
        m_pOsd->SetAttribute("display-clock", m_isClockEnabled);
        m_pOsd->SetAttribute("clock-font", m_clockFont.c_str()); 
        m_pOsd->SetAttribute("x-clock-offset", m_clockOffsetX);
        m_pOsd->SetAttribute("y-clock-offset", m_clockOffsetY);
        m_pOsd->SetAttribute("clock-font-size", m_clockFontSize);
        m_pOsd->SetAttribute("process-mode", m_processMode);
        
        SetClockColor(m_clockColorRed, m_clockColorGreen, m_clockColorBlue);
        
        AddChild(m_pQueue);
        AddChild(m_pVidConv);
        AddChild(m_pConvQueue);
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
        m_pQueue->LinkToSink(m_pVidConv);
        m_pVidConv->LinkToSink(m_pConvQueue);
        m_pConvQueue->LinkToSink(m_pOsd);
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
        m_pVidConv->UnlinkFromSink();
        m_pConvQueue->UnlinkFromSink();
        m_isLinked = false;
    }

    bool OsdBintr::AddToParent(DSL_NODETR_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' OSD to the Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
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

        *name = m_clockFont.c_str();
        *size = m_clockFontSize;
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

        m_clockColorRed = red;
        m_clockColorGreen = green;
        m_clockColorBlue = blue;
        
        uint clockColor =
          ((m_clockColorRed & 0xFF) << 24) |
          ((m_clockColorGreen & 0xFF) << 16) |
          ((m_clockColorBlue & 0xFF) << 8) | 0xFF;
              
        m_pOsd->SetAttribute("clock-color", clockColor);
        
        return true;
    }
}    