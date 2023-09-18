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

    RemuxerBintr::RemuxerBintr(const char* name, uint maxStreamIds)
        : Bintr(name)
        , m_maxStreamIds(maxStreamIds)
//        , m_streamIds(streamIds, streamIds+numStreamIds)
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
        
        // Create a new Streammuxer for this child branch with a unique name
        // based on the childs name.
        std::string streammuxerName = pChildComponent->GetName() + "-streammuxer";
        
        // We create the new Streammuxer with a Pipeline-Id of 0 so the Streammuxer
        // will not update the source-id in the frame-metadata
        DSL_MULTI_SOURCES_PTR pStreammuxerBintr = 
            DSL_MULTI_SOURCES_NEW(streammuxerName.c_str(), 0);
        
        m_muxers[streammuxerName] = pStreammuxerBintr;
        
        for (auto i=0; i<numStreamIds; i++)
        {
            std::string queueName = "source-" + std::to_string(i);
            DSL_QUEUE_SOURCE_PTR pSourceQueue = 
                DSL_QUEUE_SOURCE_NEW(queueName.c_str());
            
            pStreammuxerBintr->AddChild(
                std::dynamic_pointer_cast<SourceBintr>(pSourceQueue));
        }
        
        // Add the branch to the Remuxers collection of children branches 
        m_pChildBranches[pChildComponent->GetName()] = pChildComponent;
        
        // call the parent class to complete the add
        if (!Bintr::AddChild(pChildComponent))
        {
            LOG_ERROR("Faild to add Component '" << pChildComponent->GetName() 
                << "' as a child to '" << GetName() << "'");
            return false;
        }
        
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

        std::string streammuxerName = pChildComponent->GetName() + "-streammuxer";
        
        // Clear the Streammuxer entry from the map of child Streammuxers.
        m_muxers.erase(streammuxerName);

        // Remove the BranchBintr from the Remuxers collection of children branches
        m_pChildBranches.erase(pChildComponent->GetName());

        // call the base function to complete the remove
        return Bintr::RemoveChild(pChildComponent);
    }

    bool RemuxerBintr::IsChild(DSL_BINTR_PTR pChildComponent)
    {
        LOG_FUNC();
        
        return (m_pChildBranches.find(pChildComponent->GetName()) 
            != m_pChildBranches.end());
    }
    
    bool RemuxerBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("RemuxerBintr '" << m_name << "' is already linked");
            return false;
        }
        
//        for (auto const& ivec: m_splitters)
//        {
//            if (!ivec->LinkAll())
//            {
//                return false;
//            }
//        }
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
        
//        for (auto const& ivec: m_splitters)
//        {
//            ivec->UnlinkAll();
//        }
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