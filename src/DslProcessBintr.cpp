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
#include "DslProcessBintr.h"

namespace DSL
{
    ProcessBintr::ProcessBintr(const char* name)
        : Bintr(name)
        , m_pSinksBintr(NULL)
        , m_pOsdBintr(NULL)
    {
        LOG_FUNC();

        m_pSinksBintr = std::shared_ptr<SinksBintr>(new SinksBintr("sinks-bin"));
        
        AddChild(m_pSinksBintr);
    }

    ProcessBintr::~ProcessBintr()
    {
        LOG_FUNC();

        m_pSinksBintr = NULL;
    }
    
    void ProcessBintr::AddSinkBintr(std::shared_ptr<Bintr> pSinkBintr)
    {
        LOG_FUNC();
        
        m_pSinksBintr->AddChild(pSinkBintr);
    }
    
    void ProcessBintr::AddOsdBintr(std::shared_ptr<Bintr> pOsdBintr)
    {
        LOG_FUNC();
        
        // Add the OSD bin to this Process bin before linking to Sinks bin
        AddChild(pOsdBintr);

        m_pOsdBintr = std::dynamic_pointer_cast<OsdBintr>(pOsdBintr);

        m_pOsdBintr->LinkTo(m_pSinksBintr);
    }
    
    void ProcessBintr::AddSinkGhostPad()
    {
        LOG_FUNC();
        
        GstElement* pSinkBin;

        if (m_pOsdBintr->m_pBin)
        {
            LOG_INFO("Adding Process bin Sink Pad for OSD '" 
                << m_pOsdBintr->m_name);
            pSinkBin = m_pOsdBintr->m_pBin;
        }
        else
        {
            LOG_INFO("Adding Process bin Sink Pad for Sinks '" 
                << m_pSinksBintr->m_name);
            pSinkBin = m_pSinksBintr->m_pBin;
        }

        StaticPadtr SinkPadtr(pSinkBin, "sink");
        
        // create a new ghost pad with the Sink pad and add to this bintr's bin
        if (!gst_element_add_pad(m_pBin, gst_ghost_pad_new("sink", SinkPadtr.m_pPad)))
        {
            LOG_ERROR("Failed to add Sink Pad for '" << m_name);
        }
    }
}    
