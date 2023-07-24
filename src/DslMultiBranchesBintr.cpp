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
        : Bintr(name)
    {
        LOG_FUNC();
        
        // Single Queue and Tee element for all Components
        m_pQueue = DSL_ELEMENT_NEW("queue", name);
        m_pTee = DSL_ELEMENT_NEW(teeType, name);

        AddChild(m_pQueue);
        AddChild(m_pTee);

        // Float the Queue sink pad as a Ghost Pad for this MultiBranchesBintr
        m_pQueue->AddGhostPadToParent("sink");
        
        m_pSinkPadProbe = DSL_PAD_BUFFER_PROBE_NEW("multi-comp-sink-pad-probe", 
            "sink", m_pQueue);
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
        // Else we're adding to the end of th indexed map
        else
        {
            padId = m_usedRequestPadIds.size();
            m_usedRequestPadIds.push_back(true);
        }
        // Set the branches unique id to the available stream-id
        pChildComponent->SetRequestPadId(padId);

        // Add the branch to the Demuxers collection of children mapped by name 
        m_pChildBranches[pChildComponent->GetName()] = pChildComponent;
        
        // Add the branch to the Demuxers collection of children mapped by stream-id 
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

    typedef struct _asyncData
    {
        DslMutex asynMutex;
        DslCond asyncCond;
        GstNodetr* pChildComponent;
    } AsyncData;
    
    static GstPadProbeReturn unlink_from_source_tee_cb(GstPad* pad, 
        GstPadProbeInfo *info, gpointer pData)
    {
        AsyncData* pAsyncData = static_cast<AsyncData*>(pData);
        
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&(pAsyncData->asynMutex));
        pAsyncData->pChildComponent->UnlinkFromSourceTee();
        g_cond_signal(&(pAsyncData->asyncCond));

        return GST_PAD_PROBE_REMOVE;
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
            GstState currentState;
            GetState(currentState, 0);
            if (currentState == GST_STATE_PLAYING)
            {
                LOG_INFO("Child component '" << GetName() 
                    << "' is in a state of PLAYING - setting up async remove");

                AsyncData asyncData;
                asyncData.pChildComponent = (GstNodetr*)&*pChildComponent;
        
                LOCK_MUTEX_FOR_CURRENT_SCOPE(&asyncData.asynMutex);
                
                GstPad* pStaticSinkPad = gst_element_get_static_pad(
                    pChildComponent->GetGstElement(), "sink");
                    
                GstPad* pRequestedSrcPad = gst_pad_get_peer(pStaticSinkPad);
                    
                gst_pad_add_probe(pRequestedSrcPad, 
                    GST_PAD_PROBE_TYPE_BLOCK_DOWNSTREAM,
                    (GstPadProbeCallback)unlink_from_source_tee_cb, 
                    &asyncData, NULL);
                    
                g_cond_wait(&asyncData.asyncCond, &asyncData.asynMutex);

                gst_object_unref(pStaticSinkPad);
                gst_object_unref(pRequestedSrcPad);
            }
            else
            {
                if (!pChildComponent->UnlinkFromSourceTee())
                {   
                    LOG_ERROR("MultiBranchesBintr '" << GetName() 
                        << "' failed to Unlink Child Branch '" 
                        << pChildComponent->GetName() << "'");
                    return false;
                }                
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
            LOG_INFO("Unlinking " << m_pTee->GetName() << " from " 
                << imap.second->GetName());
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
            LOG_ERROR("Unable to set Batch size for Tee '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        for (auto const& imap: m_pChildBranches)
        {
            // unlink from the Tee Element
            if (!imap.second->SetBatchSize(batchSize))
            {
                LOG_ERROR("MultiBranchesBintr '" << GetName() 
                    << "' failed to set batch size for Child Component '" 
                    << imap.second->GetName() << "'");
                return false;
            }
        }
        return Bintr::SetBatchSize(batchSize);
    }
 
    MultiSinksBintr::MultiSinksBintr(const char* name)
        : MultiBranchesBintr(name, "tee")
    {
        LOG_FUNC();
    }
    
    SplitterBintr::SplitterBintr(const char* name)
        : MultiBranchesBintr(name, "tee")
    {
        LOG_FUNC();
    }

    bool SplitterBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' SplitterBintr to the Parent Pipeline/Branch
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddSplitterBintr(shared_from_this());
    }

    DemuxerBintr::DemuxerBintr(const char* name, uint maxBranches)
        : MultiBranchesBintr(name, "nvstreamdemux")
        , m_maxBranches(maxBranches)
    {
        LOG_FUNC();
        
        LOG_INFO("");
        LOG_INFO("Initial property values for DemuxerBintr '" << name << "'");
        LOG_INFO("  max-branches : " << m_maxBranches);
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
        // Ensure that we are not exceeding max-branches
        if ((m_pChildBranches.size()+1) > m_maxBranches)
        {
            LOG_ERROR("Can't add Branch '" << pChildComponent->GetName() 
                << "' to DemuxerBintr '" << GetName() 
                << "' as it would exceed max-branches = " << m_maxBranches);
            return false;
        }

        // find the next available unused stream-id
        uint streamId(0);
        auto ivec = find(m_usedRequestPadIds.begin(), m_usedRequestPadIds.end(), false);
        
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
            // link back upstream to the Tee - now the src for the child branch.
            if (!pChildComponent->LinkAll() or 
                !pChildComponent->LinkToSourceTee(m_pTee, 
                    m_requestedSrcPads[streamId]))
            {
                LOG_ERROR("DemuxerBintr '" << GetName() 
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

        // We need to request all the needed source pads while the 
        // nvstreamdemux plugin is in a NULL state. This is a workaround
        // for the fact the the plugin does not support dynamic requests
        for (uint i=0; i<m_maxBranches; i++)
        {
            std::string srcPadName = "src_" + std::to_string(i);
                
            GstPad* pRequestedSrcPad = gst_element_get_request_pad(
                m_pTee->GetGstElement(), srcPadName.c_str());
            if (!pRequestedSrcPad)
            {
                
                LOG_ERROR("Failed to get a requested source pad for Demuxer '" 
                    << GetName() << "'");
                return false;
            }
            LOG_INFO("Allocated requested source pad = " << pRequestedSrcPad 
                << " for DemuxerBintr '" << GetName() << "'");
            m_requestedSrcPads.push_back(pRequestedSrcPad);
        }

        for (auto const& imap: m_pChildBranchesIndexed)
        {
            // link back upstream to the Tee, the src for this Child Component 
            if (!imap.second->LinkAll() or 
                !imap.second->LinkToSourceTee(m_pTee, 
                    m_requestedSrcPads[imap.second->GetRequestPadId()]))
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
 
        // We now free all of the pre-allocated requested pads
        while (m_requestedSrcPads.size())
        {
            LOG_INFO("Releasing requested source pad = " << m_requestedSrcPads.back()
                << " for DemuxerBintr '"<< GetName() << "'");
                
            gst_element_release_request_pad(m_pTee->GetGstElement(), 
                m_requestedSrcPads.back());
            m_requestedSrcPads.pop_back();
        }
   }
   
    uint DemuxerBintr::GetMaxBranches()
    {
        LOG_FUNC();
        
        return m_maxBranches;
    }

    bool DemuxerBintr::SetMaxBranches(uint maxBranches)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't set max-branches for DemuxerBintr '" 
                << GetName() << "' as it is linked");
            return false;
        }
        if (maxBranches < m_pChildBranches.size())
        {
            LOG_ERROR("Can't set max-branches = " << maxBranches 
                << " for DemuxerBintr '" << GetName() 
                << "' as it is less than the current number of added branches");
            return false;
        }
        m_maxBranches = maxBranches;
        return true;
    }

}