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
#include "DslReporterBintr.h"
#include "DslBranchBintr.h"

namespace DSL
{

    ReporterBintr::ReporterBintr(const char* name)
        : Bintr(name)
    {
        LOG_FUNC();

        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "reporter-queue");
        
        AddChild(m_pQueue);

        m_pQueue->AddGhostPadToParent("sink");
        m_pQueue->AddGhostPadToParent("src");

        m_pSrcPadProbe = DSL_PAD_PROBE_NEW("reporter-src-pad-probe", "src", m_pQueue);
    }

    ReporterBintr::~ReporterBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }

    bool ReporterBintr::AddToParent(DSL_NODETR_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' reporter to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddReporterBintr(shared_from_this());
    }
    
    bool ReporterBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("ReporterBintr '" << m_name << "' is already linked");
            return false;
        }

        // single element, noting to link
        m_isLinked = true;
        
        return true;
    }
    
    void ReporterBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("ReporterBintr '" << m_name << "' is not linked");
            return;
        }
        // single element, nothing to unlink
        m_isLinked = false;
    }
    
}