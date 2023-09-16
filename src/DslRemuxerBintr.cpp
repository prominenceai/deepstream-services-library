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

    RemuxerBintr::RemuxerBintr(const char* name, 
        uint* streamIds, uint numStreamIds, uint maxStreamIds)
        : Bintr(name)
        , m_numStreamIds(numStreamIds)
        , m_maxStreamIds(maxStreamIds)
        , m_streamIds(streamIds, streamIds+numStreamIds)
    {
        LOG_FUNC();
        
        m_pDemuxer = DSL_DEMUXER_NEW(name, maxStreamIds);
        m_pMuxer = DSL_MULTI_SOURCES_NEW(name, 0);

        LOG_INFO("");
        LOG_INFO("Initial property values for RemuxerBintr '" << name << "'");
//        LOG_INFO("  display-bbox      : " << m_bboxEnabled);
        
        // Add each of the 
        AddChild(m_pDemuxer);
        AddChild(m_pMuxer);

        m_pDemuxer->AddGhostPadToParent("sink");
        m_pMuxer->AddGhostPadToParent("src");
    }    
    
    RemuxerBintr::~RemuxerBintr()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }

    bool RemuxerBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("RemuxerBintr '" << m_name << "' is already linked");
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
            LOG_ERROR("OsdBintr '" << m_name << "' is not linked");
            return;
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