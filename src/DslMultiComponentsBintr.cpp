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
        
        m_pSinkPadProbe = DSL_PAD_PROBE_NEW("multi-comp-sink-pad-probe", "sink", m_pQueue);
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
            if (!pChildComponent->LinkAll() or !pChildComponent->LinkToSource(m_pTee))
            {
                return false;
            }
            // Component up with the parent state
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
            // unlink the sink from the Tee
            pChildComponent->UnlinkFromSource();
            pChildComponent->UnlinkAll();
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

        uint id(0);
        for (auto const& imap: m_pChildComponents)
        {
            // Must set the Unique Id first, then Link all of the ChildComponent's Elementrs, then 
            // link back upstream to the Tee, the src for this Child Component 
            imap.second->SetId(id++);
            if (!imap.second->LinkAll() or !imap.second->LinkToSource(m_pTee))
            {
                LOG_ERROR("MultiComponentsBintr '" << GetName() 
                    << "' failed to Link Child Component '" << imap.second->GetName() << "'");
                return false;
            }
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
            if (!imap.second->UnlinkFromSource())
            {
                LOG_ERROR("MultiComponentsBintr '" << GetName() 
                    << "' failed to Unlink Child Component '" << imap.second->GetName() << "'");
                return;
            }
            // unink all of the ChildComponent's Elementrs and reset the unique Id
            imap.second->UnlinkAll();
            imap.second->SetId(-1);
        }
        m_pQueue->UnlinkFromSink();
        m_isLinked = false;
    }

    bool MultiComponentsBintr::LinkToSource(DSL_NODETR_PTR pTee)
    {
        LOG_FUNC();
        
        std::string srcPadName = "src_" + std::to_string(m_uniqueId);
        
        LOG_INFO("Linking the MultiComponentsBintr '" << GetName() << "' to Pad '" << srcPadName 
            << "' for Demuxer '" << pTee->GetName() << "'");
       
        m_pGstStaticSinkPad = gst_element_get_static_pad(GetGstElement(), "sink");
        if (!m_pGstStaticSinkPad)
        {
            LOG_ERROR("Failed to get Static Sink Pad for MuliComponentsBintr '" << GetName() << "'");
            return false;
        }

        GstPad* pGstRequestedSrcPad = gst_element_get_request_pad(pTee->GetGstElement(), srcPadName.c_str());
            
        if (!pGstRequestedSrcPad)
        {
            LOG_ERROR("Failed to get Requested Src Pad for Demuxer '" << pTee->GetName() << "'");
            return false;
        }
        m_pGstRequestedSourcePads[srcPadName] = pGstRequestedSrcPad;

        // Call the base class to complete the link relationship
        return Bintr::LinkToSource(pTee);
    }
    
    bool MultiComponentsBintr::UnlinkFromSource()
    {
        LOG_FUNC();
        
        // If we're not currently linked to the Demuxer
        if (!IsLinkedToSource())
        {
            LOG_ERROR("MultiComponentsBintr '" << GetName() << "' is not in a Linked state");
            return false;
        }

        std::string srcPadName = "src_" + std::to_string(m_uniqueId);

        LOG_INFO("Unlinking and releasing requested Source Pad for Sink Tee " << GetName());
        
        gst_pad_unlink(m_pGstRequestedSourcePads[srcPadName], m_pGstStaticSinkPad);
        gst_element_release_request_pad(GetSource()->GetGstElement(), m_pGstRequestedSourcePads[srcPadName]);
                
        m_pGstRequestedSourcePads.erase(srcPadName);
        
        return Nodetr::UnlinkFromSource();
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