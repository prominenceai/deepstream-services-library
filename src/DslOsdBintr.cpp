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
    std::string OsdBintr::m_sClockFont = "Serif";
    guint OsdBintr::m_sClockFontSize = 12;
    guint OsdBintr::m_sClockOffsetX = 800;
    guint OsdBintr::m_sClockOffsetY = 820;
    guint OsdBintr::m_sClockColor = 0;
    
    OsdBintr::OsdBintr(const char* osd, gboolean isClockEnabled)
        : Bintr(osd)
        , m_isClockEnabled(isClockEnabled)
        , m_processMode(0)
    {
        LOG_FUNC();
        
        m_pQueue = std::shared_ptr<Elementr>(new Elementr(NVDS_ELEM_QUEUE, "osd_queue", m_pBin));
        m_pVidConv = std::shared_ptr<Elementr>(new Elementr(NVDS_ELEM_VIDEO_CONV, "osd_conv", m_pBin));
        m_pConvQueue = std::shared_ptr<Elementr>(new Elementr(NVDS_ELEM_QUEUE, "osd_conv_queue", m_pBin));
        m_pOsd = std::shared_ptr<Elementr>(new Elementr(NVDS_ELEM_OSD, "nvosd0", m_pBin));

        g_object_set(G_OBJECT(m_pVidConv->m_pElement), 
            "gpu-id", m_gpuId,
            "nvbuf-memory-type", m_nvbufMemoryType, NULL);

        g_object_set(G_OBJECT(m_pOsd->m_pElement),
            "gpu-id", m_gpuId,
            "display-clock", m_isClockEnabled,
            "clock-font", (gchar*)m_sClockFont.c_str(), 
            "x-clock-offset", m_sClockOffsetX,
            "y-clock-offset", m_sClockOffsetY, 
            "clock-color", m_sClockColor,
            "clock-font-size", m_sClockFontSize, 
            "process-mode", m_processMode, NULL);

        m_pQueue->AddSinkGhostPad();
        m_pOsd->AddSourceGhostPad();
    }    
    
    OsdBintr::~OsdBintr()
    {
        LOG_FUNC();
        
    }

    void OsdBintr::LinkAll()
    {
        LOG_FUNC();
        
        m_pQueue->LinkTo(m_pVidConv);
        m_pVidConv->LinkTo(m_pConvQueue);
        m_pConvQueue->LinkTo(m_pOsd);
    }
    
    void OsdBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        m_pQueue->Unlink();
        m_pVidConv->Unlink();
        m_pConvQueue->Unlink();
    }

    void OsdBintr::AddToParent(std::shared_ptr<Bintr> pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' OSD to the Parent Pipeline 
        std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            AddOsdBintr(shared_from_this());
    }
    
}    