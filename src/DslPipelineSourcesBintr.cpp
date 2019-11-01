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

        g_object_set(m_pBin, "message-forward", TRUE, NULL);
  
        // Single Stream Muxer element for all Sources 
        m_pStreamMux = DSL_ELEMENT_NEW(NVDS_ELEM_STREAM_MUX, "stream_muxer", m_pBin);
        
        SetStreamMuxOutputSize(DSL_DEFAULT_STREAMMUX_WIDTH, DSL_DEFAULT_STREAMMUX_HEIGHT);

        // Each Source will be linked to the Stream Muxer on Source-add
        
        // Setup Src Ghost Pad for Stream Muxer element 
        AddSourceGhostPad();
    }
    
    PipelineSourcesBintr::~PipelineSourcesBintr()
    {
        LOG_FUNC();
        
        // Removed sources will be reset to not-in-use
        for (auto const& imap: m_pChildBintrs)
        {
            RemoveChild(imap.second);
        }
    }
     
    void PipelineSourcesBintr::AddChild(std::shared_ptr<Bintr> pChildBintr)
    {
        LOG_FUNC();
        
        pChildBintr->m_pParentBintr = 
            std::dynamic_pointer_cast<Bintr>(shared_from_this());
            
        // Set the play type based on the first source added
        if (m_pChildBintrs.size() == 0)
        {
            SetStreamMuxPlayType(
                std::dynamic_pointer_cast<SourceBintr>(pChildBintr)->IsLive());
        }

        m_pChildBintrs[pChildBintr->m_name] = pChildBintr;
                                
        if (!gst_bin_add(GST_BIN(m_pBin), pChildBintr->m_pBin))
        {
            LOG_ERROR("Failed to add '" << pChildBintr->m_name 
                << "' to " << m_name << "'");
            throw;
        }
    }

    void PipelineSourcesBintr::RemoveChild(std::shared_ptr<Bintr> pChildBintr)
    {
        LOG_FUNC();

        // unlink from the Streammuxer
        std::dynamic_pointer_cast<SourceBintr>(pChildBintr)->
            m_pStaticSourcePadtr->Unlink();
        
        // call the base function to complete the remove
        Bintr::RemoveChild(pChildBintr);
    }

    void PipelineSourcesBintr::RemoveAllChildren()
    {
        LOG_FUNC();
        
        // Removed sources will be reset to not-in-use
        for (auto &imap: m_pChildBintrs)
        {
            // unlink from the Streammuxer
            std::dynamic_pointer_cast<SourceBintr>(imap.second)->
                m_pStaticSourcePadtr->Unlink();
            
            // call the base function to complete the remove
            Bintr::RemoveChild(imap.second);
        }
    }

    void PipelineSourcesBintr::AddSourceGhostPad()
    {
        LOG_FUNC();
        
        // get Source pad for Stream Muxer element
        StaticPadtr sourcePadtr(m_pStreamMux->m_pElement, "src");

        // create a new ghost pad with Source pad and add to this Bintr's bin
        if (!gst_element_add_pad(m_pBin, gst_ghost_pad_new("src", sourcePadtr.m_pPad)))
        {
            LOG_ERROR("Failed to add Source Pad for '" << m_name);
            throw;
        }
        LOG_INFO("Source ghost pad added to Sources' Stream Muxer"); 
    }

    void PipelineSourcesBintr::LinkAll()
    {
        LOG_FUNC();
        
        uint id(0);
        
        for (auto const& imap: m_pChildBintrs)
        {
            std::string sinkPadName = "sink_" + std::to_string(id);
            std::dynamic_pointer_cast<SourceBintr>(imap.second)->SetSensorId(id++);
            // Retrieve the sink pad - from the Streammux element - 
            // to link to the Source component being added
            std::shared_ptr<RequestPadtr> pSinkPadtr = 
                std::shared_ptr<RequestPadtr>(new RequestPadtr(m_pStreamMux->m_pElement, (gchar*)sinkPadName.c_str()));
            
            std::dynamic_pointer_cast<SourceBintr>(imap.second)->
                m_pStaticSourcePadtr->LinkTo(pSinkPadtr);
        }
        
        // Set the Batch size to the nuber of sources owned
        // TODO add support for managing batch timeout
        SetStreamMuxBatchProperties(m_pChildBintrs.size(), 4000);
    }

    void PipelineSourcesBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        for (auto const& imap: m_pChildBintrs)
        {
            // unlink from the Streammuxer
            std::dynamic_pointer_cast<SourceBintr>(imap.second)->
                m_pStaticSourcePadtr->Unlink();
                
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
