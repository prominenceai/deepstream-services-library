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
#include "DslMultiBranchesBintr.h"
#include "DslBranchBintr.h"

namespace DSL
{

    MultiBranchesBintr::MultiBranchesBintr(const char* name, 
        const char* teeType)
        : TeeBintr(name)
    {
        LOG_FUNC();
        
        // Single Queue and Tee element for all Components
        m_pQueue = DSL_ELEMENT_NEW("queue", name);
        m_pTee = DSL_ELEMENT_NEW(teeType, name);

        AddChild(m_pQueue);
        AddChild(m_pTee);

        // Float the Queue sink pad as a Ghost Pad for this MultiBranchesBintr
        m_pQueue->AddGhostPadToParent("sink");
        
        // Add the Buffer and DS Event probes to the sink-pad of the tee element.
        AddSinkPadProbes(m_pTee);
    }
    
    MultiBranchesBintr::~MultiBranchesBintr()
    {
        LOG_FUNC();

        if (IsLinked())
        {
            UnlinkAll();
        }
    }

    bool MultiBranchesBintr::AddChild(DSL_BASE_PTR pChildElement)
    {
        LOG_FUNC();
        
        return Bintr::AddChild(pChildElement);
    }

    bool MultiBranchesBintr::AddChild(DSL_BINTR_PTR pChildComponent)
    {
        LOG_FUNC();
        
        // Ensure Component uniqueness
        if (IsChild(pChildComponent))
        {
            LOG_ERROR("'" << pChildComponent->GetName() 
                << "' is already a child of '" << GetName() << "'");
            return false;
        }
       
        // find the next available unused stream-id
        uint padId(0);
        
        // find the next available unused stream-id
        auto ivec = find(m_usedRequestPadIds.begin(), m_usedRequestPadIds.end(), false);
        
        // If we're inserting into the location of a previously remved branch
        if (ivec != m_usedRequestPadIds.end())
        {
            padId = ivec - m_usedRequestPadIds.begin();
            m_usedRequestPadIds[padId] = true;
        }
        // Else we're adding to the end of the indexed map
        else
        {
            padId = m_usedRequestPadIds.size();
            m_usedRequestPadIds.push_back(true);
        }
        // Set the branches unique id to the available stream-id
        pChildComponent->SetRequestPadId(padId);

        // Add the branch to the Tees collection of children mapped by name 
        m_pChildBranches[pChildComponent->GetName()] = pChildComponent;
        
        // Add the branch to the Tees collection of children mapped by stream-id 
        m_pChildBranchesIndexed[padId] = pChildComponent;
        
        // call the parent class to complete the add
        if (!Bintr::AddChild(pChildComponent))
        {
            LOG_ERROR("Faild to add Component '" << pChildComponent->GetName() 
                << "' as a child to '" << GetName() << "'");
            return false;
        }
        
        // If the Pipeline is currently in a linked state, linkAll Elementrs now 
        // and Link to the 
        if (IsLinked())
        {
            // link back upstream to the Tee, the src for this Child Component 
            if (!pChildComponent->LinkAll() or 
                !pChildComponent->LinkToSourceTee(m_pTee, "src_%u"))
            {
                LOG_ERROR("MultiBranchesBintr '" << GetName() 
                    << "' failed to Link Child Component '" 
                    << pChildComponent->GetName() << "'");
                return false;
            }

            // Sync component up with the parent state
            return gst_element_sync_state_with_parent(
                pChildComponent->GetGstElement());
        }
        return true;
    }
    
    bool MultiBranchesBintr::IsChild(DSL_BINTR_PTR pChildComponent)
    {
        LOG_FUNC();
        
        return (m_pChildBranches.find(pChildComponent->GetName()) 
            != m_pChildBranches.end());
    }

    bool MultiBranchesBintr::RemoveChild(DSL_BASE_PTR pChildElement)
    {
        LOG_FUNC();
        
        // call the base function to handle the remove for Elementrs
        return Bintr::RemoveChild(pChildElement);
    }

    bool MultiBranchesBintr::RemoveChild(DSL_BINTR_PTR pChildComponent)
    {
        LOG_FUNC();

        if (!IsChild(pChildComponent))
        {
            LOG_ERROR("' " << pChildComponent->GetName() 
                << "' is NOT a child of '" << GetName() << "'");
            return false;
        }
        if (pChildComponent->IsLinkedToSource())
        {  
            if (!pChildComponent->UnlinkFromSourceTee())
            {   
                LOG_ERROR("MultiBranchesBintr '" << GetName() 
                    << "' failed to Unlink Child Branch '" 
                    << pChildComponent->GetName() << "'");
                return false;
            }                
            pChildComponent->UnlinkAll();
        }
        // unreference and remove from the child-branch collections
        m_pChildBranches.erase(pChildComponent->GetName());
        m_pChildBranchesIndexed.erase(pChildComponent->GetRequestPadId());
        
        // set the used-stream id as available for reuse and clear the 
        // stream-id (id property) for the child-branch
        m_usedRequestPadIds[pChildComponent->GetRequestPadId()] = false;
        pChildComponent->SetRequestPadId(-1);
        
        // call the base function to complete the remove
        return Bintr::RemoveChild(pChildComponent);
    }

    bool MultiBranchesBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("MultiBranchesBintr '" << GetName() 
                << "' is already linked");
            return false;
        }
        m_pQueue->LinkToSink(m_pTee);

        for (auto const& imap: m_pChildBranchesIndexed)
        {
            // link back upstream to the Tee, the src for this Child Component 
            if (!imap.second->LinkAll() or 
                !imap.second->LinkToSourceTee(m_pTee, "src_%u"))
            {
                LOG_ERROR("MultiBranchesBintr '" << GetName() 
                    << "' failed to Link Child Component '" 
                    << imap.second->GetName() << "'");
                return false;
            }
        }
        m_isLinked = true;
        return true;
    }

    void MultiBranchesBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("MultiBranchesBintr '" << GetName() << "' is not linked");
            return;
        }
        for (const auto& imap: m_pChildBranchesIndexed)
        {
            // unlink from the Tee Element
            LOG_INFO("Unlinking " << m_pTee->GetName() << " from '" 
                << imap.second->GetName() << "'");
            if (!imap.second->UnlinkFromSourceTee())
            {
                LOG_ERROR("MultiBranchesBintr '" << GetName() 
                    << "' failed to Unlink Child Component '" 
                    << imap.second->GetName() << "'");
                return;
            }
            // unink all of the ChildComponent's Elementrs and reset the unique Id
            imap.second->UnlinkAll();
        }
        m_pQueue->UnlinkFromSink();
        m_isLinked = false;
    }

    bool MultiBranchesBintr::SetBatchSize(uint batchSize)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set batch-size for Tee '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        for (auto const& imap: m_pChildBranches)
        {
            // unlink from the Tee Element
            if (!imap.second->SetBatchSize(batchSize))
            {
                LOG_ERROR("MultiBranchesBintr '" << GetName() 
                    << "' failed to set batch-size for Child Component '" 
                    << imap.second->GetName() << "'");
                return false;
            }
        }
        return Bintr::SetBatchSize(batchSize);
    }
    
    //--------------------------------------------------------------------------------

    MultiSinksBintr::MultiSinksBintr(const char* name)
        : MultiBranchesBintr(name, "tee")
    {
        LOG_FUNC();

        LOG_INFO("");
        LOG_INFO("Initial property values for MultiSinksBintr '" << name << "'");
    }
    
    //--------------------------------------------------------------------------------

    SplitterBintr::SplitterBintr(const char* name)
        : MultiBranchesBintr(name, "tee")
    {
        LOG_FUNC();

        LOG_INFO("");
        LOG_INFO("Initial property values for SplitterBintr '" << name << "'");
    }

    bool SplitterBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' SplitterBintr to the Parent Pipeline/Branch
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddSplitterBintr(shared_from_this());
    }

    //-------------------------------------------------------------------------------
    
    DemuxerBintr::DemuxerBintr(const char* name)
        : MultiBranchesBintr(name, "nvstreamdemux")
    {
        LOG_FUNC();
        
        LOG_INFO("");
        LOG_INFO("Initial property values for DemuxerBintr '" << name << "'");
    }
    
    bool DemuxerBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' DemuxerBintr to the Parent Pipeline/Branch
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddDemuxerBintr(shared_from_this());
    }

    bool DemuxerBintr::AddChild(DSL_BINTR_PTR pChildComponent)
    {
        LOG_FUNC();
        
        // Ensure Component uniqueness
        if (IsChild(pChildComponent))
        {
            LOG_ERROR("'" << pChildComponent->GetName() 
                << "' is already a child of '" << GetName() << "'");
            return false;
        }

        // find the next available unused stream-id
        uint streamId(0);
        auto ivec = find(m_usedRequestPadIds.begin(), 
            m_usedRequestPadIds.end(), false);
        
        // If we're inserting into the location of a previously remved source
        if (ivec != m_usedRequestPadIds.end())
        {
            streamId = ivec - m_usedRequestPadIds.begin();
            m_usedRequestPadIds[streamId] = true;
        }
        // Else we're adding to the end of th indexed map
        else
        {
            streamId = m_usedRequestPadIds.size();
            m_usedRequestPadIds.push_back(true);
        }
        // Call the private helper to complete the common add functionality
        // now that we have the streamId
        return _completeAddChild(pChildComponent, streamId);
    }

    bool DemuxerBintr::AddChildTo(DSL_BINTR_PTR pChildComponent, uint streamId)
    {
        LOG_FUNC();
        
        // Ensure Component uniqueness
        if (IsChild(pChildComponent))
        {
            LOG_ERROR("'" << pChildComponent->GetName() 
                << "' is already a child of '" << GetName() << "'");
            return false;
        }

        // If the streamId has been used "ever", since bintr creation
        if ((streamId+1) <= m_usedRequestPadIds.size())
        {
            // Ensure that the stream-id is not currently linked
            if (m_usedRequestPadIds[streamId] == true)
            {
                LOG_ERROR("Can't add Branch '" << pChildComponent->GetName() 
                    << "' to DemuxerBintr '" << GetName() << "' at stream-id = " 
                    << streamId << " as it's currently taken");
                return false;
            }
            // Else set the used pad-ids to true at position stream-id
            else
            {
                m_usedRequestPadIds[streamId] = true;
            }
        }
        // Else, the stream-id exceeds the size so it has never been used before
        else
        {
            // Need to pad the vector with false entries up to the new
            // requested stream-id / pad-id
            for (auto i=m_usedRequestPadIds.size(); i<streamId; i++)
            {
                m_usedRequestPadIds.push_back(false);
            }
            // We can now push a true (currently used) entry at position stream-id.
            m_usedRequestPadIds.push_back(true);
        }
        // Call the private helper to complete the common add functionality
        // now that we have the streamId
        return _completeAddChild(pChildComponent, streamId);
    }
   
    bool DemuxerBintr::_completeAddChild(DSL_BINTR_PTR pChildComponent, uint streamId)
    {
        LOG_FUNC();

        // Set the branches unique id to the available stream-id
        pChildComponent->SetRequestPadId(streamId);

        // Add the branch to the Demuxers collection of children mapped by name 
        m_pChildBranches[pChildComponent->GetName()] = pChildComponent;
        
        // Add the branch to the Demuxers collection of children mapped by stream-id 
        m_pChildBranchesIndexed[streamId] = pChildComponent;
        
        // call the parent class to complete the add
        if (!Bintr::AddChild(pChildComponent))
        {
            LOG_ERROR("Failed to add Branch '" << pChildComponent->GetName() 
                << "' as a child to '" << GetName() << "'");
            return false;
        }
        
        // If the Pipeline is currently in a linked state, 
        // linkAll Elementrs now and Link with the Stream
        if (IsLinked())
        {
//            GstState currentState;
//            GetState(currentState, 0);
//            LOG_INFO("Demuxer '" << GetName() << "' is in state '" << currentState 
//                << "' while adding branch '" << pChildComponent->GetName() 
//                << "' to stream-id = " << streamId);
//                
//            if (currentState == GST_STATE_PLAYING)
//            {
//                // When in a playing state, we need to do the final sync with the
//                // parent state in the context of a PPH while blocking downstream.
//                AsyncData asyncData;
//                asyncData.pChildComponent = (GstNodetr*)&*pChildComponent;
//        
//                LOCK_MUTEX_FOR_CURRENT_SCOPE(&asyncData.asynMutex);
//
//                // IMPORTANT: we need to install the blocking probe before we link
//                // pads so that it can block the first buffer once linked.
//                gulong probeId = gst_pad_add_probe(m_requestedSrcPads[streamId], 
//                    GST_PAD_PROBE_TYPE_BLOCK_DOWNSTREAM,
//                    (GstPadProbeCallback)link_to_source_tee_cb, 
//                    &asyncData, NULL);
//
//                // link back upstream to the Tee - now the src for the child branch.
//                if (!pChildComponent->LinkAll() or 
//                    !pChildComponent->LinkToSourceTee(m_pTee, 
//                        m_requestedSrcPads[streamId]))
//                {
//                    LOG_ERROR("DemuxerBintr '" << GetName() 
//                        << "' failed to Link Child Component '" 
//                        << pChildComponent->GetName() << "'");
//                    return false;
//                }
//
//                gint64 endTime = g_get_monotonic_time() + (G_TIME_SPAN_SECOND *
//                    m_blockingTimeout);
//                    
//                if (!g_cond_wait_until(&asyncData.asyncCond, 
//                    &asyncData.asynMutex, endTime))
//                {
//                    // timeout - individual source must be paused or not linked.
//                    
//                    LOG_ERROR("Timout waiting for blocking pad probe adding branch '" 
//                        << pChildComponent->GetName() << "' to Parent Demuxer");
//                    LOG_ERROR("Upstream source must be in a non-playing state");
//                    
//                    pChildComponent->UnlinkFromSourceTee();
//                    pChildComponent->UnlinkAll();
//                    // remove the probe since it timed out.
//                    gst_pad_remove_probe(m_requestedSrcPads[streamId], probeId);
//                    return false;
//                }
//                else
//                {
//                    // branch has been succcessfully added while the Pipeline 
//                    // and individual source stream were both playing.
//                    return true;
//                }
//            }
//            else
//            {
            std::string srcPadName = "src_" + 
                std::to_string(pChildComponent->GetRequestPadId());
    
            if (!pChildComponent->LinkAll() or 
                !pChildComponent->LinkToSourceTee(m_pTee, 
                    srcPadName.c_str()))
            {
                LOG_ERROR("DemuxerBintr '" << GetName() 
                    << "' failed to Link Child Component '" 
                    << pChildComponent->GetName() << "'");
                return false;
            }
            // Sync the branch with the parent (this demuxer) state now.
            LOG_INFO("Synchronizing branch '" << pChildComponent->GetName() 
                << "' with Parent Demuxer");
            
            return gst_element_sync_state_with_parent(
                pChildComponent->GetGstElement());
        }
        return true;
    }
    
    bool DemuxerBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("DemuxerBintr '" << GetName() 
                << "' is already linked");
            return false;
        }
        m_pQueue->LinkToSink(m_pTee);

        for (auto const& imap: m_pChildBranchesIndexed)
        {
            std::string srcPadName = "src_" + 
                std::to_string(imap.second->GetRequestPadId());

            // link back upstream to the Tee, the src for this Child Component 
            if (!imap.second->LinkAll() or 
                !imap.second->LinkToSourceTee(m_pTee, 
                    srcPadName.c_str()))
            {
                LOG_ERROR("DemuxerBintr '" << GetName() 
                    << "' failed to Link Child Component '" 
                        << imap.second->GetName() << "'");
                return false;
            }
        }
        m_isLinked = true;
        return true;
    }

    void DemuxerBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("DemuxerBintr '" << GetName() << "' is not linked");
            return;
        }

        // Call the parent class to do the actual unlinking
        MultiBranchesBintr::UnlinkAll();
   }
   
}