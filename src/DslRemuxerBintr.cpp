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
#include "DslRemuxerBintr.h"
#include "DslPipelineBintr.h"

namespace DSL
{
    //--------------------------------------------------------------------------------
    
    RemuxerQueueBintr::RemuxerQueueBintr(const char* name)
        : Bintr(name)
    {
        LOG_FUNC();

        m_pQueue = DSL_ELEMENT_NEW("queue", name);

        LOG_INFO("");
        LOG_INFO("Initial property values for RemuxerQueueBintr '" << name << "'");
        
        AddChild(m_pQueue);

        // Queue is both "sink" and "src" ghost-pad for this Bintr.
        m_pQueue->AddGhostPadToParent("sink");
        m_pQueue->AddGhostPadToParent("src");
    }
    
    RemuxerQueueBintr::~RemuxerQueueBintr()
    {
        LOG_FUNC();

    }

    bool RemuxerQueueBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("RemuxerQueueBintr '" << GetName() 
                << "' is already in a linked state");
            return false;
        }
        m_isLinked = true;
        
        return true;
    }

    void RemuxerQueueBintr::UnlinkAll()
    {
        LOG_FUNC();

        if (!m_isLinked)
        {
            LOG_ERROR("RemuxerQueueBintr '" << GetName() 
                << "' is not in a linked state");
            return;
        }
        m_isLinked = false;
    }
    
    //--------------------------------------------------------------------------------

    RemuxerBranchBintr::RemuxerBranchBintr(const char* name, 
        GstObject* parentRemuxerBin, 
        const std::vector<DSL_SPLITTER_PTR>& splitters,
        DSL_BINTR_PTR pChildBranch, uint* streamIds, uint numStreamIds)
        : Bintr(name, parentRemuxerBin)
        , m_pChildBranch(pChildBranch)
        , m_streamIds(streamIds, streamIds+numStreamIds)
    {
        LOG_FUNC();
        
        // Create a new Streammuxer for this child branch with a unique name
        // derived from this RemuxerBranchBintr's name which comes from the
        // unique pChildBranch name - enforced by the client API.
        std::string streammuxerName = GetName() + "-streammuxer";
        
        // Each branch needs its own Streammuxer to connect back to the output Tees,
        // which are connected to the request-pads of the Demuxer.
        // We create the new Streammuxer with a pipeline-id of -1 so the Streammuxer
        // will not update the source-ids in the frame-metadata.
        m_pStreammuxerBintr = DSL_MULTI_SOURCES_NEW(streammuxerName.c_str(), -1);
       
        // Add both the ChildBranch and new Streammuxer to this RemuxerBranchBintr
        // so that we can link them together => streammuxer->branch.
        AddChild(m_pChildBranch);
        AddChild(m_pStreammuxerBintr);

        // Next, we loop through the stream-ids vector to create the required 
        // Queue and Identity bintrs used to link up the Demuxer output Tees
        // to the branch's Streammuxer.
        //
        // Demuxer[stream-i]->Tee[pad-id]->Queue->Identity->[stream-i]Streammuxer
        //
        // The first Branch added will link to each Tee on request-pad-id = 0.
        // The second Branch added will link to each Tee on request-pad-d = 1.
        for (auto i=0; i<m_streamIds.size(); i++)
        {
            // Create the IdentitySource that will link upstream with the Queue
            // and downstream with the Streammuxer
            std::string sourceName = GetName() +"-source-" + 
                std::to_string(m_streamIds[i]);
                
            DSL_IDENTITY_SOURCE_PTR pIdentitySource = 
                DSL_IDENTITY_SOURCE_NEW(sourceName.c_str());
            
            // Add the IdentitySource as a Child of the Streammuxer to be linked
            // to the ith-requested sink-pad. 
            m_pStreammuxerBintr->AddChild(
                std::dynamic_pointer_cast<SourceBintr>(pIdentitySource));

            // We need to create a ghost pad for the added Identity Source
            // so that it can be linked with the Queue - which is not a child.
            // This elevates the pad to the same level as the Streammuxer.
            // The pad will be named from the unique Stream-id it is connecting to.
            std::string ghostPadName = "sink_" + std::to_string(m_streamIds[i]);
                
            if (!gst_element_add_pad(m_pStreammuxerBintr->GetGstElement(), 
                gst_ghost_pad_new(ghostPadName.c_str(), 
                    gst_element_get_static_pad(pIdentitySource->GetGstElement(),
                        "sink"))))
            {
                LOG_ERROR("Failed to add ghost-pad '" << ghostPadName 
                    << "' for queue-source'" << pIdentitySource->GetName() << "'");
                throw;
            }
            
            // Create the Queue that will link upstream with the Splitter Tee
            // and downstream with the Queue
            std::string queueName = GetName() +"-queue-" + 
                std::to_string(m_streamIds[i]);
                
            DSL_REMUXER_QUEUE_PTR pQueueBintr = 
                DSL_REMUXER_QUEUE_NEW(queueName.c_str());
            
            // Add the Queue as a Child of the Tee that is connected the the
            // Demuxer's requested source-pad for the current stream-id in the vector. 
            splitters[m_streamIds[i]]->AddChild(
                std::dynamic_pointer_cast<Bintr>(pQueueBintr));

            // We need to create a ghost pad for the added Queue so that
            // it can be linked with the Indenty Source - which is not a child.
            // This elevates the pad to the same level as the Tee.
            // The pad is named from the pad-id of the Tee it will connect to. 
            ghostPadName = "src_" + 
                std::to_string(pQueueBintr->GetRequestPadId());
                
            if (!gst_element_add_pad(splitters[m_streamIds[i]]->GetGstElement(), 
                gst_ghost_pad_new(ghostPadName.c_str(), 
                    gst_element_get_static_pad(pQueueBintr->GetGstElement(),
                        "src"))))
            {
                LOG_ERROR("Failed to add ghost-pad '" << ghostPadName 
                    << "' for queue-source'" << pQueueBintr->GetName() << "'");
                throw;
            }
            
            // Add the created IdentitySource and Queue their respective colections.
            m_identiySources[m_streamIds[i]] = pIdentitySource;
            m_queueBintrs[m_streamIds[i]] = pQueueBintr;
        }
        
    }
    
    RemuxerBranchBintr::~RemuxerBranchBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }

    bool RemuxerBranchBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("RemuxerBranchBintr '" << GetName() 
                << "' is already linked");
            return false;
        }
        
        if (!m_pChildBranch->LinkAll() or
            !m_pStreammuxerBintr->LinkAll() or 
            !m_pStreammuxerBintr->LinkToSink(m_pChildBranch))
        {
            return false;
        }
        m_isLinked = true;
        return true;
    }
    
    void RemuxerBranchBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("RemuxerBranchBintr '" << GetName() 
                << "' is not linked");
            return;
        }
        m_pStreammuxerBintr->UnlinkFromSink();
        m_pStreammuxerBintr->UnlinkAll();
        m_pChildBranch->UnlinkAll();

        m_isLinked = false;
    }

    bool RemuxerBranchBintr::LinkToSourceTees(
        const std::vector<DSL_SPLITTER_PTR>& splitters)
    {
        LOG_FUNC();
    
        // We loop through the stream-ids vector to link each Tee with
        // with the Streammuxer ... by linking the Queue (via ghostpad)
        // with the IdentitySource (via ghostpad)
        for (auto i=0; i<m_streamIds.size(); i++)
        {
            // IMPORTANT! these names need to match the pad-names created when the
            // RemuxerBranchBintr, Queue's and IdentitySource's were created.
            std::string srcPadName = "src_" + 
                std::to_string(m_queueBintrs[m_streamIds[i]]->GetRequestPadId());
                
            std::string sinkPadName = "sink_" + std::to_string(m_streamIds[i]);

            if (!gst_element_link_pads(
                splitters[m_streamIds[i]]->GetGstElement(), srcPadName.c_str(),
                m_pStreammuxerBintr->GetGstElement(), sinkPadName.c_str()))
            {
                LOG_ERROR("Failed to link '" << m_queueBintrs[m_streamIds[i]]->GetName() 
                    << "' to '" << m_identiySources[m_streamIds[i]]->GetName() << "'");
                return false;
            }   
            LOG_INFO("Successfully linked '" << m_queueBintrs[m_streamIds[i]]->GetName() 
                << "' to '" << m_identiySources[m_streamIds[i]]->GetName() << "'");
        }
        return true;
    }
    
    void RemuxerBranchBintr::UnlinkFromSourceTees(
        const std::vector<DSL_SPLITTER_PTR>& splitters)
    {
        LOG_FUNC();
    
        // We loop through the stream-ids vector to unlink each Tee from
        // the Streammuxer ... by unlinking the Queue (via ghostpad)
        // from the IdentitySource (via ghostpad)
        for (auto i=0; i<m_streamIds.size(); i++)
        {
            // IMPORTANT! these names need to match the pad-names created when the
            // RemuxerBranchBintr, Queue's and IdentitySource's were created.
            std::string srcPadName = "src_" + 
                std::to_string(m_queueBintrs[m_streamIds[i]]->GetRequestPadId());
                
            std::string sinkPadName = "sink_" + std::to_string(m_streamIds[i]);

            gst_element_unlink_pads(
                splitters[m_streamIds[i]]->GetGstElement(), srcPadName.c_str(),
                m_pStreammuxerBintr->GetGstElement(), sinkPadName.c_str());

            LOG_INFO("Successfully unlinked '" << m_queueBintrs[m_streamIds[i]]->GetName() 
                << "' to '" << m_identiySources[m_streamIds[i]]->GetName() << "'");
        }
    }

    void RemuxerBranchBintr::GetStreammuxBatchProperties(uint* batchSize, 
        int* batchTimeout)
    {
        LOG_FUNC();

        m_pStreammuxerBintr->
            GetStreammuxBatchProperties(batchSize, batchTimeout);
    }

    bool RemuxerBranchBintr::SetStreammuxBatchProperties(uint batchSize, 
        int batchTimeout)
    {
        LOG_FUNC();

        return m_pStreammuxerBintr->
            SetStreammuxBatchProperties(batchSize, batchTimeout);
    }

    void RemuxerBranchBintr::GetStreammuxDimensions(uint* width, uint* height)
    {
        LOG_FUNC();

        m_pStreammuxerBintr->GetStreammuxDimensions(width, height);
    }

    bool RemuxerBranchBintr::SetStreammuxDimensions(uint width, uint height)
    {
        LOG_FUNC();

        return m_pStreammuxerBintr->SetStreammuxDimensions(width, height);
    }
    
    //--------------------------------------------------------------------------------
            
    RemuxerBintr::RemuxerBintr(const char* name, uint maxStreamIds)
        : Bintr(name)
        , m_maxStreamIds(maxStreamIds)
        , m_streammuxWidth(DSL_STREAMMUX_DEFAULT_WIDTH)
        , m_streammuxHeight(DSL_STREAMMUX_DEFAULT_HEIGHT)
        , m_batchTimeout(-1)
    {
        LOG_FUNC();
        
        m_pDemuxer = DSL_DEMUXER_NEW(name, maxStreamIds);
        
        // Create a splitter Tee for each "potential" demuxed source stream.
        // Each tee will be linked to a unique Demuxer requested source pad.
        for (auto i=0; i<maxStreamIds; i++)
        {
            // Create the unique name for the splitter based on stream-id
            std::string splitterName = GetName() + "-splitter-" + std::to_string(i);
            
            // Create a new splitter and add as a child to the Demuxer.
            DSL_SPLITTER_PTR pSplitter = DSL_SPLITTER_NEW(splitterName.c_str());
            m_pDemuxer->AddChildTo(pSplitter, i);

            // We also add the Splitter to the vector of splitters for link access.
            m_splitters.push_back(pSplitter);
            
            // Create the default stream-ids for all possible stream-ids.
            // This will be used for Branches that are simply added, without
            // specifying a select set of stream-ids as with "add-to"
            m_defaultStreamIds.push_back(i);
        }

        LOG_INFO("");
        LOG_INFO("Initial property values for RemuxerBintr '" << name << "'");
        LOG_INFO("  width                  : " << m_streammuxWidth);
        LOG_INFO("  height                 : " << m_streammuxHeight);
        LOG_INFO("  batched-push-timeout   : " << m_batchTimeout);
        
        // Add the demuxer as child and elevate as sink ghost pad
        Bintr::AddChild(m_pDemuxer);

        m_pDemuxer->AddGhostPadToParent("sink");
    }    
    
    RemuxerBintr::~RemuxerBintr()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }
    
    bool RemuxerBintr::AddChild(DSL_BINTR_PTR pChildComponent)
    {
        LOG_FUNC();

        if (IsChild(pChildComponent))
        {
            LOG_ERROR("Component '" << pChildComponent->GetName() 
                << "' is already a child of '" << GetName() << "'");
            return false;
        }
        
        // Important - use the child branch components name. 
        // use the default-stream-ids meaning all possible stream-ids.
        // The Branches are created as proxy-bins for this bintr.
        DSL_REMUXER_BRANCH_PTR pRemuxerBranch = 
            DSL_REMUXER_BRANCH_NEW(pChildComponent->GetCStrName(),
                GetGstObject(), m_splitters, pChildComponent, 
                &m_defaultStreamIds[0], m_defaultStreamIds.size());
        
        // Add the branch to the Remuxers collection of children branches 
        m_childBranches[pChildComponent->GetName()] = pRemuxerBranch;
        
        return true;
    }
    
    bool RemuxerBintr::AddChildTo(DSL_BINTR_PTR pChildComponent, 
        uint* streamIds, uint numStreamIds)
    {
        LOG_FUNC();

        if (IsChild(pChildComponent))
        {
            LOG_ERROR("Component '" << pChildComponent->GetName() 
                << "' is already a child of '" << GetName() << "'");
            return false;
        }
        
        // Important - use the child branch components name. 
        // The Branches are created as proxy-bins for this bintr.
        DSL_REMUXER_BRANCH_PTR pRemuxerBranch = 
            DSL_REMUXER_BRANCH_NEW(pChildComponent->GetCStrName(),
                GetGstObject(), m_splitters, 
                    pChildComponent, streamIds, numStreamIds);
        
        // Add the branch to the Remuxers collection of children branches 
        m_childBranches[pChildComponent->GetName()] = pRemuxerBranch;
        
        return true;
    }
    
    bool RemuxerBintr::RemoveChild(DSL_BINTR_PTR pChildComponent)
    {
        LOG_FUNC();

        if (!IsChild(pChildComponent))
        {
            LOG_ERROR("Component '" << pChildComponent->GetName() 
                << "' is not a child of '" << GetName() << "'");
            return false;
        }

        // Remove the BranchBintr from the Remuxers collection of children branches.
        // This will destroy
        m_childBranches.erase(pChildComponent->GetName());
        
        return true;
    }

    bool RemuxerBintr::IsChild(DSL_BINTR_PTR pChildComponent)
    {
        LOG_FUNC();
        
        return (m_childBranches.find(pChildComponent->GetName()) 
            != m_childBranches.end());
    }
    
    bool RemuxerBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("RemuxerBintr '" << m_name << "' is already linked");
            return false;
        }
        
        if (!m_pDemuxer->LinkAll())
        {
            return false;
        }
        for (auto const& imap: m_childBranches)
        {
            int batchTimeout(0); // we don't care about batch-timeout
            imap.second->GetStreammuxBatchProperties(&mbatchSize, &batchTimeout);
            if (!imap.second->LinkAll() or
                !imap.second->LinkToSourceTees(m_splitters))
            {
                return false;
            }
        }

        m_isLinked = true;
        return true;
    }
    
    void RemuxerBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("RemuxerBintr '" << m_name << "' is not linked");
            return;
        }
        m_pDemuxer->UnlinkAll();
        
        for (auto const& imap: m_childBranches)
        {
            imap.second->UnlinkAll();
            imap.second->UnlinkFromSourceTees(m_splitters);
        }
        m_isLinked = false;
    }

    bool RemuxerBintr::AddToParent(DSL_BASE_PTR pBranchBintr)
    {
        LOG_FUNC();
        
        // add 'this' OSD to the Parent Pipeline 
//        return std::dynamic_pointer_cast<BranchBintr>(pBranchBintr)->
//            AddRemuxerBintr(shared_from_this());
        return true;
    }

    bool RemuxerBintr::RemoveFromParent(DSL_BASE_PTR pBranchBintr)
    {
        LOG_FUNC();
        
        // remove 'this' OSD to the Parent Pipeline 
//        return std::dynamic_pointer_cast<BranchBintr>(pBranchBintr)->
//            RemoveRemuxerBintr(shared_from_this());
        return true;
    }

    int* RemuxerBintr::GetBatchTimeout(int* batchTimeout)
    {
        LOG_FUNC();
        
        return m_batchTimeout;
    }

    bool RemuxerBintr::SetBatchTimeout(int batchTimeout)
    {
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't set Streammuxer dimension for RemuxerBintr '" 
                << GetName << "' as it is currently linked");
            return false;
        }
        m_batchTimeout = batchTimeout;

        for (auto const& imap: m_childBranches)
        {
            if (!imap.second->SetBatchTimeout(m_batchTimeout))
            {
                return false;
            }
        }
        return true;
    }
    
    void RemuxerBranchBintr::GetStreammuxDimensions(uint* width, uint* height)
    {
        LOG_FUNC();

        *width = m_streammuxWidth;
        *height = m_streammuxHeight;
    }

    bool RemuxerBranchBintr::SetStreammuxDimensions(uint width, uint height)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't set Streammuxer dimension for RemuxerBintr '" 
                << GetName << "' as it is currently linked");
            return false;
        }
        m_streammuxWidth = width;
        m_streammuxHeight = height;

        for (auto const& imap: m_childBranches)
        {
            if (!imap.second->SetStreammuxDimensions(
                m_streammuxWidth, m_streammuxHeight))
            {
                return false;
            }
        }
        return true;
    }

    bool RemuxerBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Unable to set GPU ID for OsdBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }

        m_gpuId = gpuId;

        return true;
    }
}