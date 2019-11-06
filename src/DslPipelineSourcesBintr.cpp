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
#include "DslApi.h"
#include "DslPipelineSourcesBintr.h"

namespace DSL
{
    PipelineSourcesBintr::PipelineSourcesBintr(const char* name)
        : Bintr(name)
    {
        LOG_FUNC();

        g_object_set(m_pGstObj, "message-forward", TRUE, NULL);
  
        // Single Stream Muxer element for all Sources 
        m_pStreamMux = DSL_ELEMENT_NEW(NVDS_ELEM_STREAM_MUX, "stream_muxer");
        
        SetStreamMuxOutputSize(DSL_DEFAULT_STREAMMUX_WIDTH, DSL_DEFAULT_STREAMMUX_HEIGHT);

        // Setup Src Ghost Pad for Stream Muxer element 
        m_pStreamMux->AddGhostPad("src");
        
        AddChild(m_pStreamMux);
    }
    
    PipelineSourcesBintr::~PipelineSourcesBintr()
    {
        LOG_FUNC();
        
        // Removed sources will be reset to not-in-use
        RemoveAllChildren();
    }
     
    DSL_NODETR_PTR PipelineSourcesBintr::AddChild(DSL_NODETR_PTR pChildBintr)
    {
        LOG_FUNC();
        
        // Set the play type based on the first source added
        if (m_pChildren.size() == 0)
        {
            SetStreamMuxPlayType(
                std::dynamic_pointer_cast<SourceBintr>(pChildBintr)->IsLive());
        }
                                
        if (!gst_bin_add(GST_BIN(m_pGstObj), GST_ELEMENT(pChildBintr->m_pGstObj)))
        {
            LOG_ERROR("Failed to add '" << pChildBintr->m_name 
                << "' to " << m_name << "'");
            throw;
        }
        return Bintr::AddChild(pChildBintr);
    }

    void PipelineSourcesBintr::RemoveChild(DSL_NODETR_PTR pChildBintr)
    {
        LOG_FUNC();

        // unlink from the Streammuxer
        std::dynamic_pointer_cast<SourceBintr>(pChildBintr)->
            m_pStaticPadtr->Unlink();
        
        // call the base function to complete the remove
        Bintr::RemoveChild(pChildBintr);
    }

    void PipelineSourcesBintr::RemoveAllChildren()
    {
        LOG_FUNC();
        
        // Removed sources will be reset to not-in-use
        for (auto &imap: m_pChildren)
        {
            // unlink from the Streammuxer
            std::dynamic_pointer_cast<SourceBintr>(imap.second)->
                m_pStaticPadtr->Unlink();
            
            // call the base function to complete the remove
            Bintr::RemoveChild(imap.second);
        }
    }


    bool PipelineSourcesBintr::LinkAll()
    {
        LOG_FUNC();
        
        uint id(0);
        
        for (auto const& imap: m_pChildren)
        {
            std::string sinkPadName = "sink_" + std::to_string(id);
            std::dynamic_pointer_cast<SourceBintr>(imap.second)->SetSensorId(id++);
            
            // Retrieve the sink pad - from the Streammux element - 
            // to link to the Source component being added
            DSL_REQUEST_PADTR_PTR pSinkPadtr = 
                DSL_REQUEST_PADTR_NEW(sinkPadName.c_str(), m_pStreamMux);
            
            std::dynamic_pointer_cast<SourceBintr>(imap.second)->
                m_pStaticPadtr->LinkTo(pSinkPadtr);
        }
        
        // Set the Batch size to the nuber of sources owned
        // TODO add support for managing batch timeout
        SetStreamMuxBatchProperties(m_pChildren.size(), 4000);
        
        return true;
    }

    void PipelineSourcesBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        for (auto const& imap: m_pChildren)
        {
            // unlink from the Streammuxer
//            std::dynamic_pointer_cast<SourceBintr>(imap.second)->
//                m_pStaticSourcePadtr->Unlink();
                
            // reset the Sensor ID to unlinked     
            std::dynamic_pointer_cast<SourceBintr>(imap.second)->SetSensorId(-1);
        }
    }

    void PipelineSourcesBintr::SetStreamMuxPlayType(bool areSourcesLive)
    {
        LOG_FUNC();
        
        m_areSourcesLive = areSourcesLive;
        
        m_pStreamMux->SetAttribute("live-source", m_areSourcesLive);
        
    }

    void PipelineSourcesBintr::SetStreamMuxBatchProperties(uint batchSize, uint batchTimeout)
    {
        LOG_FUNC();
        
        m_batchSize = batchSize;
        m_batchTimeout = batchTimeout;

        m_pStreamMux->SetAttribute("batch-size", m_batchSize);
        m_pStreamMux->SetAttribute("batched-push-timeout", m_batchTimeout);
    }

    void PipelineSourcesBintr::SetStreamMuxOutputSize(uint width, uint height)
    {
        LOG_FUNC();

        m_streamMuxWidth = width;
        m_streamMuxHeight = height;

        m_pStreamMux->SetAttribute("width", m_streamMuxWidth);
        m_pStreamMux->SetAttribute("height", m_streamMuxHeight);
    }
}
