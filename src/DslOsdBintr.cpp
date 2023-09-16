/*
The MIT License

Copyright (c) 2019-2023, Prominence AI, Inc.

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

    OsdBintr::OsdBintr(const char* name, 
        boolean textEnabled, boolean clockEnabled,
        boolean bboxEnabled, boolean maskEnabled)
        : Bintr(name)
        , m_textEnabled(textEnabled)
        , m_clockEnabled(clockEnabled)
        , m_bboxEnabled(bboxEnabled)
        , m_maskEnabled(maskEnabled)
        , m_processMode(DSL_OSD_PROCESS_MODE_CPU)
        , m_clockFont(DSL_DEFAULT_OSD_CLOCK_FONT_TYPE)
        , m_clockFontSize(DSL_DEFAULT_OSD_CLOCK_FONT_SIZE)
        , m_clockOffsetX(DSL_DEFAULT_OSD_CLOCK_OFFSET_X)
        , m_clockOffsetY(DSL_DEFAULT_OSD_CLOCK_OFFSET_Y)
        , m_clockColor(DSL_DEFAULT_OSD_CLOCK_COLOR)
    {
        LOG_FUNC();
        
        // Create all elements
        m_pVidConvQueue = DSL_ELEMENT_EXT_NEW("queue", name, "nvvideoconvert");
        m_pVidConv = DSL_ELEMENT_NEW("nvvideoconvert", name);
        m_pOsdQueue = DSL_ELEMENT_EXT_NEW("queue", name, "nvdsosd");
        m_pOsd = DSL_ELEMENT_NEW("nvdsosd", name);

        // Get property defaults that aren't specifically set
        m_pVidConv->GetAttribute("gpu-id", &m_gpuId);
        m_pVidConv->GetAttribute("nvbuf-memory-type", &m_nvbufMemType);

        // Override the default process-mode
        m_pOsd->SetAttribute("process-mode", m_processMode);

        // Set the enabled flags - from our input parameters
        m_pOsd->SetAttribute("display-clock", m_clockEnabled);
        m_pOsd->SetAttribute("display-text", m_textEnabled);
        m_pOsd->SetAttribute("display-bbox", m_bboxEnabled);
        m_pOsd->SetAttribute("display-mask", m_maskEnabled);

        // Override the default nvdsosd plugin clock settings
        m_pOsd->SetAttribute("clock-font", m_clockFont.c_str()); 
        m_pOsd->SetAttribute("clock-font-size", m_clockFontSize);
        m_pOsd->SetAttribute("x-clock-offset", m_clockOffsetX);
        m_pOsd->SetAttribute("y-clock-offset", m_clockOffsetY);
        
        // Call the SetClockColor method to set the m_clkRgbaColor variable
        // which is then used to set the and nvdsosd "clock-color" property
        SetClockColor(m_clockColor.red, m_clockColor.green,
            m_clockColor.blue, m_clockColor.alpha);
        
        LOG_INFO("");
        LOG_INFO("Initial property values for OsdBintr '" << name << "'");
        LOG_INFO("  display-bbox      : " << m_bboxEnabled);
        LOG_INFO("  display-text      : " << m_textEnabled);
        LOG_INFO("  display-mask      : " << m_maskEnabled);
        LOG_INFO("  display-clock     : " << m_clockEnabled);
        LOG_INFO("  clock-font        : " << m_clockFont);
        LOG_INFO("  clock-font-size   : " << m_clockFontSize);
        LOG_INFO("  clock-color       : " << m_clkRgbaColor);
        LOG_INFO("  clock-x-offset    : " << m_clockOffsetX);
        LOG_INFO("  clock-y-offset    : " << m_clockOffsetY);
        LOG_INFO("  process-mode      : " << m_processMode);
        LOG_INFO("  gpu-id            : " << m_gpuId);
        LOG_INFO("  nvbuf-memory-type : " << m_nvbufMemType);
        
        // Add each of the 
        AddChild(m_pVidConvQueue);
        AddChild(m_pVidConv);
        AddChild(m_pOsdQueue);
        AddChild(m_pOsd);

        m_pVidConvQueue->AddGhostPadToParent("sink");
        m_pOsd->AddGhostPadToParent("src");

        m_pSinkPadProbe = DSL_PAD_BUFFER_PROBE_NEW("osd-sink-pad-probe", "sink", m_pOsdQueue);
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
        
        if (!m_pVidConvQueue->LinkToSink(m_pVidConv) or
            !m_pVidConv->LinkToSink(m_pOsdQueue) or
            !m_pOsdQueue->LinkToSink(m_pOsd))
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
        m_pVidConvQueue->UnlinkFromSink();
        m_pVidConv->UnlinkFromSink();
        m_pOsdQueue->UnlinkFromSink();
        m_isLinked = false;
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
        
        // OSD plugin supports dynamic updates to "display-text"
        // do not check link-state

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
        
        // OSD plugin supports dynamic updates to "display-clock"
        // do not check link-state

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
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set clock offsets for OsdBintr '" << GetName() 
                << "' as it's currently linked");
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
            LOG_ERROR("Unable to set clock font for OsdBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }

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
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set clock color for OsdBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }

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

    void OsdBintr::GetBboxEnabled(boolean* enabled)
    {
        LOG_FUNC();
        
        *enabled = m_bboxEnabled;
    }
    
    bool OsdBintr::SetBboxEnabled(boolean enabled)
    {
        LOG_FUNC();
        
        // OSD plugin supports dynamic updates to "display-bbox"
        // do not check link-state

        m_bboxEnabled = enabled;
        m_pOsd->SetAttribute("display-bbox", m_bboxEnabled);
        
        return true;
    }

    void OsdBintr::GetMaskEnabled(boolean* enabled)
    {
        LOG_FUNC();
        
        *enabled = m_maskEnabled;
    }
    
    bool OsdBintr::SetMaskEnabled(boolean enabled)
    {
        LOG_FUNC();

        // OSD plugin supports dynamic updates to "display-mask"
        // do not check link-state
        
        m_maskEnabled = enabled;
        m_pOsd->SetAttribute("display-mask", m_maskEnabled);
        
        return true;
    }

    void OsdBintr::GetProcessMode(uint* mode)
    {
        LOG_FUNC();
        
        *mode = m_processMode;
    }
    
    bool OsdBintr::SetProcessMode(uint mode)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set the process-mode setting for OsdBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_processMode = mode;
        m_pOsd->SetAttribute("process-mode", m_processMode);
        
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

        m_pVidConv->SetAttribute("gpu-id", m_gpuId);
        m_pOsd->SetAttribute("gpu-id", m_gpuId);
        
        LOG_INFO("SecondaryGieBintr '" << GetName() 
            << "' - new GPU ID = " << m_gpuId );

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
        m_pVidConv->SetAttribute("nvbuf-memory-type", m_nvbufMemType);

        return true;
    }
}    