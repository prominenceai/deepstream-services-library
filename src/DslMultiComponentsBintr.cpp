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
#include "DslMultiComponentsBintr.h"
#include "DslBranchBintr.h"

namespace DSL
{

    MultiComponentsBintr::MultiComponentsBintr(const char* name, const char* teeType)
        : Bintr(name)
    {
        LOG_FUNC();
        
        // Single Queue and Tee element for all Components
        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "sink_bin_queue");
        m_pTee = DSL_ELEMENT_NEW(teeType, "sink_bin_tee");

        AddChild(m_pQueue);
        AddChild(m_pTee);

        // Float the Queue sink pad as a Ghost Pad for this MultiComponentsBintr
        m_pQueue->AddGhostPadToParent("sink");
        
        m_pSinkPadProbe = DSL_PAD_BUFFER_PROBE_NEW("multi-comp-sink-pad-probe", "sink", m_pQueue);
    }
    
    MultiComponentsBintr::~MultiComponentsBintr()
    {
        LOG_FUNC();

        if (IsLinked())
        {
            UnlinkAll();
        }
    }

    bool MultiComponentsBintr::AddChild(DSL_BASE_PTR pChildElement)
    {
        LOG_FUNC();
        
        return Bintr::AddChild(pChildElement);
    }

    bool MultiComponentsBintr::AddChild(DSL_BINTR_PTR pChildComponent)
    {
        LOG_FUNC();
        
        // Ensure Component uniqueness
        if (IsChild(pChildComponent))
        {
            LOG_ERROR("'" << pChildComponent->GetName() << "' is already a child of '" << GetName() << "'");
            return false;
        }

        // Add the Component to the Components collection and as a child of this Bintr
        m_pChildComponents[pChildComponent->GetName()] = pChildComponent;
        
        // call the base function to complete the add
        if (!Bintr::AddChild(pChildComponent))
        {
            LOG_ERROR("Faild to add Component '" << pChildComponent->GetName() << "' as a child to '" << GetName() << "'");
            return false;
        }
        
        // If the Pipeline is currently in a linked state, Set child source Id to the next available,
        // linkAll Elementrs now and Link to with the Stream
        if (IsLinked())
        {
            uint streamId(0);
            
            // find the next available unused stream-id
            auto ivec = find(m_usedStreamIds.begin(), m_usedStreamIds.end(), false);
            if (ivec != m_usedStreamIds.end())
            {
                streamId = ivec - m_usedStreamIds.begin();
                m_usedStreamIds[streamId] = true;
            }
            else
            {
                streamId = m_usedStreamIds.size();
                m_usedStreamIds.push_back(true);
            }
            // Must set the Unique Id first, then Link all of the ChildComponent's Elementrs, then 
            pChildComponent->SetId(streamId);

            // NOTE: important to use the correct request pad name based on the element type
            // Cast the base DSL_BASE_PTR to DSL_ELEMENTR_PTR so we can query the factory type 
            DSL_ELEMENT_PTR pTeeElementr = 
                std::dynamic_pointer_cast<Elementr>(m_pTee);

             std::string srcPadName = (pTeeElementr->IsFactoryName("nvstreamdemux"))
                ? "src_" + std::to_string(streamId)
                : "src_%u";
                
            // link back upstream to the Tee, the src for this Child Component 
            if (!pChildComponent->LinkAll() or !pChildComponent->LinkToSourceTee(m_pTee, srcPadName.c_str()))
            {
                LOG_ERROR("MultiComponentsBintr '" << GetName() 
                    << "' failed to Link Child Component '" << pChildComponent->GetName() << "'");
                return false;
            }

            // Sync component up with the parent state
            return gst_element_sync_state_with_parent(pChildComponent->GetGstElement());
        }
        return true;
    }
    
    bool MultiComponentsBintr::IsChild(DSL_BINTR_PTR pChildComponent)
    {
        LOG_FUNC();
        
        return (m_pChildComponents.find(pChildComponent->GetName()) != m_pChildComponents.end());
    }

    bool MultiComponentsBintr::RemoveChild(DSL_BASE_PTR pChildElement)
    {
        LOG_FUNC();
        
        // call the base function to handle the remove for Elementrs
        return Bintr::RemoveChild(pChildElement);
    }

    bool MultiComponentsBintr::RemoveChild(DSL_BINTR_PTR pChildComponent)
    {
        LOG_FUNC();

        if (!IsChild(pChildComponent))
        {
            LOG_ERROR("' " << pChildComponent->GetName() << "' is NOT a child of '" << GetName() << "'");
            return false;
        }
        if (pChildComponent->IsLinkedToSource())
        {
            pChildComponent->UnlinkFromSourceTee();
            pChildComponent->UnlinkAll();
            
            // set the used-stream id as available for reuse
            m_usedStreamIds[pChildComponent->GetId()] = false;
        }
        
        // unreference and remove from the collection of sinks
        m_pChildComponents.erase(pChildComponent->GetName());
        
        // call the base function to complete the remove
        return Bintr::RemoveChild(pChildComponent);
    }


    bool MultiComponentsBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("MultiComponentsBintr '" << GetName() << "' is already linked");
            return false;
        }
        m_pQueue->LinkToSink(m_pTee);

        uint streamId(0);
        for (auto const& imap: m_pChildComponents)
        {
            // Must set the Unique Id first, then Link all of the ChildComponent's Elementrs, then 
            imap.second->SetId(streamId);

            // NOTE: important to use the correct request pad name based on the element type
            // Cast the base DSL_BASE_PTR to DSL_ELEMENTR_PTR so we can query the factory type 
            DSL_ELEMENT_PTR pTeeElementr = 
                std::dynamic_pointer_cast<Elementr>(m_pTee);

             std::string srcPadName = (pTeeElementr->IsFactoryName("nvstreamdemux"))
                ? "src_" + std::to_string(streamId)
                : "src_%u";
                
            // link back upstream to the Tee, the src for this Child Component 
            if (!imap.second->LinkAll() or !imap.second->LinkToSourceTee(m_pTee, srcPadName.c_str()))
            {
                LOG_ERROR("MultiComponentsBintr '" << GetName() 
                    << "' failed to Link Child Component '" << imap.second->GetName() << "'");
                return false;
            }
            // add the new stream id to the vector of currently connected (used) 
            m_usedStreamIds.push_back(true);
            streamId++;
        }
        m_isLinked = true;
        return true;
    }

    void MultiComponentsBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("MultiComponentsBintr '" << GetName() << "' is not linked");
            return;
        }
        for (auto const& imap: m_pChildComponents)
        {
            // unlink from the Tee Element
            LOG_INFO("Unlinking " << m_pTee->GetName() << " from " << imap.second->GetName());
            if (!imap.second->UnlinkFromSourceTee())
            {
                LOG_ERROR("MultiComponentsBintr '" << GetName() 
                    << "' failed to Unlink Child Component '" << imap.second->GetName() << "'");
                return;
            }
            // unink all of the ChildComponent's Elementrs and reset the unique Id
            imap.second->UnlinkAll();
            imap.second->SetId(-1);
        }
        m_usedStreamIds.clear();
        m_pQueue->UnlinkFromSink();
        m_isLinked = false;
    }

    bool MultiComponentsBintr::SetBatchSize(uint batchSize)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Batch size for Tee '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        for (auto const& imap: m_pChildComponents)
        {
            // unlink from the Tee Element
            if (!imap.second->SetBatchSize(batchSize))
            {
                LOG_ERROR("MultiComponentsBintr '" << GetName() 
                    << "' failed to set batch size for Child Component '" << imap.second->GetName() << "'");
                return false;
            }
        }
        return Bintr::SetBatchSize(batchSize);
    }
 
    MultiSinksBintr::MultiSinksBintr(const char* name)
        : MultiComponentsBintr(name, "tee")
    {
        LOG_FUNC();
    }
    
    DemuxerBintr::DemuxerBintr(const char* name)
        : MultiComponentsBintr(name, "nvstreamdemux")
    {
        LOG_FUNC();
    }

    bool DemuxerBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' tiler to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddDemuxerBintr(shared_from_this());
    }
   
    SplitterBintr::SplitterBintr(const char* name)
        : MultiComponentsBintr(name, "tee")
    {
        LOG_FUNC();
    }

    bool SplitterBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' tiler to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddSplitterBintr(shared_from_this());
    }

}