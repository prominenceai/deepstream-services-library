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
    {
        LOG_FUNC();

        // Single Queue and Tee element for all Sinks
        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "sink_bin_queue");
        m_pTee = DSL_ELEMENT_NEW(NVDS_ELEM_TEE, "sink_bin_tee");
        
        AddChild(m_pQueue);
        AddChild(m_pTee);

        // Float the Queue sink pad as a Ghost Pad for this PipelineSinksBintr
        m_pQueue->AddGhostPadToParent("sink");
    }
    
    PipelineSinksBintr::~PipelineSinksBintr()
    {
        LOG_FUNC();

        if (IsLinked())
        {
            UnlinkAll();
        }
    }
     
    bool PipelineSinksBintr::AddChild(DSL_NODETR_PTR pChildElement)
    {
        LOG_FUNC();
        
        return Bintr::AddChild(pChildElement);
    }

    bool PipelineSinksBintr::AddChild(DSL_SINK_PTR pChildSink)
    {
        LOG_FUNC();
        
        // Ensure Sink uniqueness
        if (IsChild(pChildSink))
        {
            LOG_ERROR("' " << pChildSink->GetName() << "' is already a child of '" << GetName() << "'");
            return false;
        }
        
        // Add the Sink to the Sinks collection and as a child of this Bintr
        m_pChildSinks[pChildSink->GetName()] = pChildSink;
        return Bintr::AddChild(pChildSink);
    }
    
    bool PipelineSinksBintr::IsChild(DSL_SINK_PTR pChildSink)
    {
        LOG_FUNC();
        
        return (bool)m_pChildSinks[pChildSink->GetName()];
    }

    bool PipelineSinksBintr::RemoveChild(DSL_NODETR_PTR pChildElement)
    {
        LOG_FUNC();
        
        // call the base function to handle the remove for Elementrs
        return Bintr::RemoveChild(pChildElement);
    }

    bool PipelineSinksBintr::RemoveChild(DSL_SINK_PTR pChildSink)
    {
        LOG_FUNC();

        if (!IsChild(pChildSink))
        {
            LOG_ERROR("' " << pChildSink->GetName() << "' is NOT a child of '" << GetName() << "'");
            throw;
        }
        if (pChildSink->IsLinkedToSource())
        {
            // unlink the sink from the Tee
            pChildSink->UnlinkFromSource();
        }
        
        // unreference and remove from the collection of sinks
        m_pChildSinks.erase(pChildSink->GetName());
        
        // call the base function to complete the remove
        return Bintr::RemoveChild(pChildSink);
    }


    bool PipelineSinksBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("PipelineSinksBintr '" << GetName() << "' is already linked");
            return false;
        }
        m_pQueue->LinkToSink(m_pTee);
        
        uint id(0);
        for (auto const& imap: m_pChildSinks)
        {
            // Must set the Unique Id first, then Link all of the ChildSink's Elementrs, then 
            // link back upstream to the Tee, the src for this Child Sink 
            imap.second->SetSinkId(id++);
            if (!imap.second->LinkAll() or !imap.second->LinkToSource(m_pTee))
            {
                LOG_ERROR("PipelineSinksBintr '" << GetName() 
                    << "' failed to Link Child Sink '" << imap.second->GetName() << "'");
                return false;
            }
        }
        m_isLinked = true;
        return true;
    }

    void PipelineSinksBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("OsdBintr '" << GetName() << "' is not linked");
            return;
        }
        for (auto const& imap: m_pChildSinks)
        {
            // unlink from the Tee Element
            LOG_INFO("Unlinking " << m_pTee->GetName() << " from " << imap.second->GetName());
            if (!imap.second->UnlinkFromSource())
            {
                LOG_ERROR("PipelineSinksBintr '" << GetName() 
                    << "' failed to Unlink Child Sink '" << imap.second->GetName() << "'");
                return;
            }
            // unink all of the ChildSink's Elementrs and reset the unique Id
            imap.second->UnlinkAll();
            imap.second->SetSinkId(-1);
        }
        m_pQueue->UnlinkFromSink();
        m_isLinked = false;
    }
}