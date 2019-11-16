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

        AddChild(m_pStreamMux);

        // Float the StreamMux src pad as a Ghost Pad for this PipelineSourcesBintr
        m_pStreamMux->AddGhostPadToParent("src");
    }
    
    PipelineSourcesBintr::~PipelineSourcesBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
        if (IsLinkedToSink())
        {
            UnlinkFromSink();
        }
        if (IsLinkedToSource())
        {
            UnlinkFromSource();
        }

        if (m_pGstSinkPad)
        {
            LOG_INFO("Unreferencing GST Sink Pad for SourcesBintr '" << GetName() << "'");
            
            gst_object_unref(m_pGstSinkPad);
            m_pGstSinkPad = NULL;
        }
        if (m_pGstSourcePad)
        {
            LOG_INFO("Unreferencing GST Source Pad for SourcesBintr '" << GetName() << "'");
            
            gst_object_unref(m_pGstSourcePad);
            m_pGstSourcePad = NULL;
        }

        // Remove all child references 
        RemoveAllChildren();
        
        if (m_pGstObj and !m_pParentGstObj and (GST_OBJECT_REFCOUNT_VALUE(m_pGstObj) == 1))
        {
            LOG_INFO("Unreferencing GST Object contained by this Bintr '" << GetName() << "'");
            
            gst_object_unref(m_pGstObj);
        }
        LOG_INFO("Nodetr '" << GetName() << "' deleted");
    }

    bool PipelineSourcesBintr::AddChild(DSL_NODETR_PTR pChildElement)
    {
        LOG_FUNC();
        
        return Bintr::AddChild(pChildElement);
    }
     
    bool PipelineSourcesBintr::AddChild(DSL_SOURCE_PTR pChildSource)
    {
        LOG_FUNC();
        
        // Ensure source uniqueness
        if (IsChild(pChildSource))
        {
            LOG_ERROR("Source '" << pChildSource->GetName() << "' is already a child of '" << GetName() << "'");
            return false;
        }
        
        // Set the play type based on the first source added
        if (m_pChildSources.size() == 0)
        {
            SetStreamMuxPlayType(
                std::dynamic_pointer_cast<SourceBintr>(pChildSource)->IsLive());
        }
        
        // Add the Source to the Sources collection and as a child of this Bintr
        m_pChildSources[pChildSource->GetName()] = pChildSource;
        return Bintr::AddChild(pChildSource);
    }

    bool PipelineSourcesBintr::IsChild(DSL_SOURCE_PTR pChildSource)
    {
        LOG_FUNC();
        
        return (bool)m_pChildSources[pChildSource->GetName()];
    }

    bool PipelineSourcesBintr::RemoveChild(DSL_NODETR_PTR pChildElement)
    {
        LOG_FUNC();
        
        // call the base function to handle the remove for Elementrs
        return Bintr::RemoveChild(pChildElement);
    }

    bool PipelineSourcesBintr::RemoveChild(DSL_SOURCE_PTR pChildSource)
    {
        LOG_FUNC();

        // Check for the relationship first
        if (!IsChild(pChildSource))
        {
            LOG_ERROR("Source '" << pChildSource->GetName() << "' is not a child of '" << GetName() << "'");
            throw;
        }

        // unlink the source from the Streammuxer
        pChildSource->UnlinkFromSink();
        
        // unreference and remove from the collection of source
        m_pChildSources.erase(pChildSource->GetName());
        
        // call the base function to complete the remove
        return Bintr::RemoveChild(pChildSource);
    }

    bool PipelineSourcesBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("PipelineSourcesBintr '" << GetName() << "' is already linked");
            return false;
        }
        uint id(0);
        
        for (auto const& imap: m_pChildSources)
        {
            imap.second->SetSensorId(id++);
            imap.second->LinkAll();
            imap.second->LinkToSink(m_pStreamMux);
        }
        m_isLinked = true;

        // Set the Batch size to the nuber of sources owned
        // TODO add support for managing batch timeout
        SetStreamMuxBatchProperties(m_pChildSources.size(), 4000);
        
        return true;
    }

    void PipelineSourcesBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("PipelineSourcesBintr '" << GetName() << "' is not linked");
            return;
        }
        for (auto const& imap: m_pChildSources)
        {
            imap.second->UnlinkFromSink();
        }
        m_isLinked = false;
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

        LOG_INFO("Setting StreamMux batch properties: batch-size = " << m_batchSize 
            << ", batch-timeout = " << m_batchTimeout);

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
