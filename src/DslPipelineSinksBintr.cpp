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
#include "DslPipelineSinksBintr.h"

namespace DSL
{

    PipelineSinksBintr::PipelineSinksBintr(const char* name)
        : Bintr(name)
        , m_pQueue(NULL)
        , m_pTee(NULL)
    {
        LOG_FUNC();

        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "sink_bin_queue", m_pBin);
        m_pTee = DSL_ELEMENT_NEW(NVDS_ELEM_TEE, "sink_bin_tee", m_pBin);
        
        m_pQueue->AddSinkGhostPad();
    }
    
    PipelineSinksBintr::~PipelineSinksBintr()
    {
        LOG_FUNC();
    }
     
    void PipelineSinksBintr::AddChild(std::shared_ptr<Bintr> pChildBintr)
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
        
        pSourcePadtr->LinkTo(std::dynamic_pointer_cast<OverlaySinkBintr>(pChildBintr)->m_pStaticSinkPadtr);
    };
}    
