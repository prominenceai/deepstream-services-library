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
//    std::string OsdBintr::m_sClockFont = "Serif";
//    guint OsdBintr::m_sClockFontSize = 12;
//    guint OsdBintr::m_sClockOffsetX = 800;
//    guint OsdBintr::m_sClockOffsetY = 820;
//    guint OsdBintr::m_sClockColor = 0;
    
    OsdBintr::OsdBintr(const char* osd, gboolean isClockEnabled)
        : Bintr(osd)
        , m_isClockEnabled(isClockEnabled)
        , m_processMode(0)
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
        m_pOsd->SetAttribute("clock-font", m_sClockFont.c_str()); 
        m_pOsd->SetAttribute("x-clock-offset", m_sClockOffsetX);
        m_pOsd->SetAttribute("y-clock-offset", m_sClockOffsetY);
        m_pOsd->SetAttribute("clock-color", m_sClockColor);
        m_pOsd->SetAttribute("clock-font-size", m_sClockFontSize);
        m_pOsd->SetAttribute("process-mode", m_processMode);

        m_pQueue->AddGhostPad("sink");
        m_pOsd->AddGhostPad("src");
        
        AddChild(m_pQueue);
        AddChild(m_pVidConv);
        AddChild(m_pVidConv);
        AddChild(m_pOsd);
    }    
    
    OsdBintr::~OsdBintr()
    {
        LOG_FUNC();
        
    }

    bool OsdBintr::LinkAll()
    {
        LOG_FUNC();
        
        m_pQueue->LinkTo(m_pVidConv);
        m_pVidConv->LinkTo(m_pConvQueue);
        m_pConvQueue->LinkTo(m_pOsd);
        
        return true;
    }
    
    void OsdBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        m_pQueue->Unlink();
        m_pVidConv->Unlink();
        m_pConvQueue->Unlink();
    }

    void OsdBintr::AddToParent(DSL_NODETR_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' OSD to the Parent Pipeline 
        std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            AddOsdBintr(shared_from_this());
    }
    
}    