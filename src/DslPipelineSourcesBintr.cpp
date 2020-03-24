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
        , m_batchSize(0)
        , m_batchTimeout(0)
        , m_streamMuxWidth(0)
        , m_streamMuxHeight(0)
        , m_isPaddingEnabled(false)
        , m_areSourcesLive(false)
    {
        LOG_FUNC();

        g_object_set(m_pGstObj, "message-forward", TRUE, NULL);
  
        // Single Stream Muxer element for all Sources 
        m_pStreamMux = DSL_ELEMENT_NEW(NVDS_ELEM_STREAM_MUX, "stream_muxer");
        
        SetStreamMuxDimensions(DSL_DEFAULT_STREAMMUX_WIDTH, DSL_DEFAULT_STREAMMUX_HEIGHT);

        AddChild(m_pStreamMux);

        // Float the StreamMux src pad as a Ghost Pad for this PipelineSourcesBintr
        m_pStreamMux->AddGhostPadToParent("src");
    }
    
    PipelineSourcesBintr::~PipelineSourcesBintr()
    {
        LOG_FUNC();

        if (IsLinked())
        {
            UnlinkAll();
        }
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
        
        if (!Bintr::AddChild(pChildSource))
        {
            LOG_ERROR("Faild to add Source '" << pChildSource->GetName() << "' as a child to '" << GetName() << "'");
            return false;
        }
        
        // If the Pipeline is currently in a linked state, Set child source Id to the next available,
        // linkAll Elementrs now and Link to with the Stream
        if (IsLinked())
        {
            pChildSource->SetId(GetNumChildren() - 1);
            if (!pChildSource->LinkAll() or !pChildSource->LinkToSink(m_pStreamMux))
            {
                return false;
            }
            // Sink up with the parent state
            return gst_element_sync_state_with_parent(pChildSource->GetGstElement());
        }
        return true;
        
    }

    bool PipelineSourcesBintr::IsChild(DSL_SOURCE_PTR pChildSource)
    {
        LOG_FUNC();
        
        return (m_pChildSources.find(pChildSource->GetName()) != m_pChildSources.end());
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
            return false;
        }

        if (pChildSource->IsLinkedToSink())
        {
            // unlink the source from the Streammuxer
            pChildSource->UnlinkFromSink();
            pChildSource->UnlinkAll();
        }
        
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
            // Must set the Unique Id first, then Link all of the ChildSources's Elementrs, then 
            // link back downstream to the StreamMux, the sink for this Child Souce 
            imap.second->SetId(id++);
            if (!imap.second->LinkAll() or !imap.second->LinkToSink(m_pStreamMux))
            {
                LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                    << "' failed to Link Child Source '" << imap.second->GetName() << "'");
                return false;
            }
        }
        if (!m_batchSize)
        {
            // Set the Batch size to the nuber of sources owned if not already set
            // TODO add support for managing batch timeout
            SetStreamMuxBatchProperties(m_pChildSources.size(), 40000);
        }
        m_isLinked = true;
        
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
            // unlink from the Tee Element
            LOG_INFO("Unlinking " << m_pStreamMux->GetName() << " from " << imap.second->GetName());
            if (!imap.second->UnlinkFromSink())
            {
                LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                    << "' failed to Unlink Child Source '" << imap.second->GetName() << "'");
                return;
            }
            // unink all of the ChildSource's Elementrs and reset the unique Id
            imap.second->UnlinkAll();
            imap.second->SetId(-1);

        }
        m_isLinked = false;
    }
    
    void PipelineSourcesBintr::SetStreamMuxPlayType(bool areSourcesLive)
    {
        LOG_FUNC();
        
        m_areSourcesLive = areSourcesLive;
        
        m_pStreamMux->SetAttribute("live-source", m_areSourcesLive);
    }

    bool PipelineSourcesBintr::StreamMuxPlayTypeIsLive()
    {
        LOG_FUNC();
        
        return m_areSourcesLive;
    }
    
    void PipelineSourcesBintr::GetStreamMuxBatchProperties(guint* batchSize, guint* batchTimeout)
    {
        LOG_FUNC();

        *batchSize = m_batchSize;
        *batchTimeout = m_batchTimeout;
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
    
    void PipelineSourcesBintr::GetStreamMuxDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_streamMuxWidth;
        *height = m_streamMuxHeight;
    }

    void PipelineSourcesBintr::SetStreamMuxDimensions(uint width, uint height)
    {
        LOG_FUNC();

        m_streamMuxWidth = width;
        m_streamMuxHeight = height;

        LOG_INFO("Setting StreamMux dimensions: width = " << m_streamMuxWidth 
            << ", height = " << m_streamMuxWidth);

        m_pStreamMux->SetAttribute("width", m_streamMuxWidth);
        m_pStreamMux->SetAttribute("height", m_streamMuxHeight);
    }
    
    void PipelineSourcesBintr::GetStreamMuxPadding(bool* enabled)
    {
        LOG_FUNC();
        
        *enabled = m_isPaddingEnabled;
    }
    
    void PipelineSourcesBintr::SetStreamMuxPadding(bool enabled)
    {
        LOG_FUNC();
        
        m_isPaddingEnabled = enabled;
        
        LOG_INFO("Setting StreamMux attribute: enable-padding = " << m_isPaddingEnabled); 
        
        m_pStreamMux->SetAttribute("enable-padding", m_isPaddingEnabled);
    }
}
