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

    RemuxerBranchBintr::RemuxerBranchBintr(const char* name, 
        GstObject* parentRemuxerBin, DSL_BINTR_PTR pChildBranch,
        uint* streamIds, uint numStreamIds)
        : Bintr(name, parentRemuxerBin)
        , m_pChildBranch(pChildBranch)
        , m_streamIds(streamIds, streamIds+numStreamIds)
    {
        LOG_FUNC();
        
        // Create a new Streammuxer for this child branch with a unique name
        std::string streammuxerName = GetName() + "-streammuxer";
        
        // We create the new Streammuxer with a pipeline-id of -1 so the Streammuxer
        // will not update the source-id in the frame-metadata
        m_pStreammuxerBintr = DSL_MULTI_SOURCES_NEW(streammuxerName.c_str(), -1);
       
        AddChild(m_pChildBranch);
        AddChild(m_pStreammuxerBintr);

        for (auto i=0; i<numStreamIds; i++)
        {
            std::string queueName = GetName() +"-source-" + std::to_string(i);
            DSL_QUEUE_SOURCE_PTR pQueueSource = 
                DSL_QUEUE_SOURCE_NEW(queueName.c_str());
            
            m_pStreammuxerBintr->AddChild(
                std::dynamic_pointer_cast<SourceBintr>(pQueueSource));
                
            m_queueSources[m_streamIds[i]] = pQueueSource;
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

            
    RemuxerBintr::RemuxerBintr(const char* name, uint maxStreamIds)
        : Bintr(name)
        , m_maxStreamIds(maxStreamIds)
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

            // We also add the Splitter to vector of all splitters for link access.
            m_splitters.push_back(pSplitter);
        }

        LOG_INFO("");
        LOG_INFO("Initial property values for RemuxerBintr '" << name << "'");
//        LOG_INFO("  display-bbox      : " << m_bboxEnabled);
        
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
    
    bool RemuxerBintr::AddChild(DSL_BINTR_PTR pChildComponent, 
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
        DSL_REMUXER_BRANCH_PTR pRemuxerBranch = 
            DSL_REMUXER_BRANCH_NEW(pChildComponent->GetCStrName(),
                GetGstObject(), pChildComponent, streamIds, numStreamIds);
        
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
        
        for (auto const& imap: m_childBranches)
        {
            if (!imap.second->LinkAll())
            {
                return false;
            }
        }
        if (!m_pDemuxer->LinkAll())
        {
            return false;
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