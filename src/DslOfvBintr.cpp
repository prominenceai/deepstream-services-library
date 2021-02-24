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
#include "DslOfvBintr.h"
#include "DslPipelineBintr.h"

namespace DSL
{

    OfvBintr::OfvBintr(const char* name)
        : Bintr(name)
    {
        LOG_FUNC();
        
        m_pSinkQueue = DSL_ELEMENT_NEW("queue", "opt-flow-sink-queue");
        m_pOptFlow = DSL_ELEMENT_NEW("nvof", "opt-flow");
        m_pOptFlowQueue = DSL_ELEMENT_NEW("queue", "opt-flow-visual-queue");
        m_pOptFlowVisual = DSL_ELEMENT_NEW("nvofvisual", "opt-flow-visual");

        
        AddChild(m_pSinkQueue);
        AddChild(m_pOptFlow);
        AddChild(m_pOptFlowQueue);
        AddChild(m_pOptFlowVisual);

        m_pSinkQueue->AddGhostPadToParent("sink");
        m_pOptFlowVisual->AddGhostPadToParent("src");

        m_pSinkPadProbe = DSL_PAD_BUFFER_PROBE_NEW("osd-sink-pad-probe", "sink", m_pSinkQueue);
        m_pSrcPadProbe = DSL_PAD_BUFFER_PROBE_NEW("osd-src-pad-probe", "src", m_pOptFlowVisual);
    }    
    
    OfvBintr::~OfvBintr()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }

    bool OfvBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("OfvBintr '" << m_name << "' is already linked");
            return false;
        }
        if (!m_pSinkQueue->LinkToSink(m_pOptFlow) or
            !m_pOptFlow->LinkToSink(m_pOptFlowQueue) or
            !m_pOptFlowQueue->LinkToSink(m_pOptFlowVisual))
        {
            return false;
        }
        m_isLinked = true;
        return true;
    }
    
    void OfvBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("OfvBintr '" << m_name << "' is not linked");
            return;
        }
        m_pSinkQueue->UnlinkFromSink();
        m_pOptFlow->UnlinkFromSink();
        m_pOptFlowQueue->UnlinkFromSink();
        m_isLinked = false;
    }
    
    bool OfvBintr::AddToParent(DSL_BASE_PTR pBranchBintr)
    {
        LOG_FUNC();
        
        // add 'this' OSD to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pBranchBintr)->
            AddOfvBintr(shared_from_this());
    }
    
}    