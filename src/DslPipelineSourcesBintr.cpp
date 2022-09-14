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
#include "DslApi.h"
#include "DslPadProbeHandler.h"
#include "DslPipelineSourcesBintr.h"
#include "DslServices.h"

namespace DSL
{
    PipelineSourcesBintr::PipelineSourcesBintr(const char* name)
        : Bintr(name)
        , m_isPaddingEnabled(false)
        , m_areSourcesLive(false)
        , m_nvbufMemType(DSL_STREAMMUX_DEFAULT_NVBUF_MEMORY_TYPE)
        , m_batchTimeout(DSL_STREAMMUX_DEFAULT_BATCH_TIMEOUT)
    {
        LOG_FUNC();

        g_object_set(m_pGstObj, "message-forward", TRUE, NULL);
  
        // Single Stream Muxer element for all Sources 
        m_pStreamMux = DSL_ELEMENT_NEW("nvstreammux", name);

        // Setup all default properties
        //SetStreamMuxNvbufMemType(DSL_STREAMMUX_DEFAULT_NVBUF_MEMORY_TYPE);
        SetStreamMuxDimensions(DSL_STREAMMUX_DEFAULT_WIDTH, DSL_STREAMMUX_DEFAULT_HEIGHT);
        SetStreamMuxNumSurfacesPerFrame(DSL_STREAMMUX_DEFAULT_MAX_NUM_SERFACES_PER_FRAME);

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

    bool PipelineSourcesBintr::AddChild(DSL_BASE_PTR pChildElement)
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
            LOG_ERROR("Source '" << pChildSource->GetName() 
                << "' is already a child of '" << GetName() << "'");
            return false;
        }
        
        // Set the play type based on the first source added
        if (m_pChildSources.size() == 0)
        {
            StreamMuxPlayTypeIsLiveSet(pChildSource->IsLive());
        }
        else if (pChildSource->IsLive() != StreamMuxPlayTypeIsLiveGet())
        {
            LOG_ERROR("Can't add Source '" << pChildSource->GetName() 
                << "' with IsLive=" << pChildSource->IsLive()  << " to streamuxer '" 
                << GetName() << "' with IsLive=" << StreamMuxPlayTypeIsLiveGet());
            return false;
        }
        // If we're adding an RTSP Source, determine if EOS consumer 
        // should be added to Streammuxer
        if (pChildSource->IsType(typeid(RtspSourceBintr)) 
            and m_pEosConsumer == nullptr)
        {
            DSL_RTSP_SOURCE_PTR pRtspSource = 
                std::dynamic_pointer_cast<RtspSourceBintr>(pChildSource);
            
            // If stream management is enabled for at least one RTSP source, 
            // add the EOS Consumer
            if (pRtspSource->GetBufferTimeout())
            {
                LOG_INFO("Adding EOS Consumer to Streammuxer 'src' pad on first RTSP Source");
                
                // Create the Pad Probe and EOS Consumer to drop the EOS event that occurs on 
                // loss of RTSP stream, allowing the Pipeline to continue to play. Each RTSP source 
                // will then manage their own restart attempts and time management.

                std::string eventHandlerName = GetName() + "-eos-consumer";
                m_pEosConsumer = DSL_PPEH_EOS_CONSUMER_NEW(eventHandlerName.c_str());

                std::string padProbeName = GetName() + "-src-pad-probe";
                m_pSrcPadProbe = DSL_PAD_EVENT_DOWNSTREAM_PROBE_NEW(padProbeName.c_str(), 
                    "src", m_pStreamMux);
                m_pSrcPadProbe->AddPadProbeHandler(m_pEosConsumer);
            }
        }
        
        // Add the Source to the Sources collection and as a child of this Bintr
        m_pChildSources[pChildSource->GetName()] = pChildSource;
        
        if (!Bintr::AddChild(pChildSource))
        {
            LOG_ERROR("Faild to add Source '" << pChildSource->GetName() 
                << "' as a child to '" << GetName() << "'");
            return false;
        }
        
        // If the Pipeline is currently in a linked state, Set child source 
        // Id to the next available, linkAll Elementrs now and Link to the Stream-muxwwer
        if (IsLinked())
        {
           uint padId(0);
            
            // find the next available unused stream-id
            auto ivec = find(m_usedPadIds.begin(), m_usedPadIds.end(), false);
            if (ivec != m_usedPadIds.end())
            {
                padId = ivec - m_usedPadIds.begin();
                m_usedPadIds[padId] = true;
            }
            else
            {
                padId = m_usedPadIds.size();
                m_usedPadIds.push_back(true);
            }            
            std::string sinkPadName = "sink_" + std::to_string(padId);
            
            if (!pChildSource->LinkAll() or 
                !pChildSource->LinkToSinkMuxer(m_pStreamMux, sinkPadName.c_str()))
            {
                LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                    << "' failed to Link Child Source '" << pChildSource->GetName() << "'");
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

    bool PipelineSourcesBintr::RemoveChild(DSL_BASE_PTR pChildElement)
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
            LOG_ERROR("Source '" << pChildSource->GetName() 
                << "' is not a child of '" << GetName() << "'");
            return false;
        }

        if (IsLinked())
        {
            LOG_INFO("Unlinking " << m_pStreamMux->GetName() << " from " 
                << pChildSource->GetName());
                
            // unlink the source from the Streammuxer
            if (!pChildSource->UnlinkFromSinkMuxer())
            {   
                LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                    << "' failed to Unlink Child Source '" << pChildSource->GetName() << "'");
                return false;
            }
            // unlink all of the ChildSource's Elementrs
            pChildSource->UnlinkAll();
        }
        
        // unreference and remove from the collection of sources
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
        
        uint padId(0);
        
        for (auto const& imap: m_pChildSources)
        {
            std::string sinkPadName = "sink_" + std::to_string(padId);
            
            if (!imap.second->LinkAll() or !imap.second->LinkToSinkMuxer(m_pStreamMux,
                sinkPadName.c_str()))
            {
                LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                    << "' failed to Link Child Source '" << imap.second->GetName() << "'");
                return false;
            }
            // add the new stream id to the vector of currently connected (used) 
            m_usedPadIds.push_back(true);
            padId++;
        }
        // Set the Batch size to the nuber of sources owned if not already set
        if (!m_batchSize)
        {
            m_batchSize = m_pChildSources.size();
            m_pStreamMux->SetAttribute("batch-size", m_batchSize);
            m_pStreamMux->SetAttribute("batched-push-timeout", m_batchTimeout);
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
            // unlink from the Streammuxer
            LOG_INFO("Unlinking " << m_pStreamMux->GetName() 
                << " from " << imap.second->GetName());
            if (!imap.second->UnlinkFromSinkMuxer())
            {   
                LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                    << "' failed to Unlink Child Source '" << imap.second->GetName() << "'");
                return;
            }
            
            // unlink all of the ChildSource's Elementrs
            imap.second->UnlinkAll();
        }

        m_usedPadIds.clear();
        m_isLinked = false;
    }
    
    void PipelineSourcesBintr::EosAll()
    {
        LOG_FUNC();
        
        // Send EOS message to each source object.
        for (auto const& imap: m_pChildSources)
        {
//            LOG_INFO("Send EOS message to Source "  << imap.second->GetName());
            gst_element_send_event(imap.second->GetGstElement(), 
                gst_event_new_eos());
        }
    }
    
    
    bool PipelineSourcesBintr::StreamMuxPlayTypeIsLiveGet()
    {
        LOG_FUNC();
        
        return m_areSourcesLive;
    }
    
    void PipelineSourcesBintr::StreamMuxPlayTypeIsLiveSet(bool isLive)
    {
        LOG_FUNC();
        
        m_areSourcesLive = isLive;
        
        LOG_INFO("'live-source' attrubute set to '" << m_areSourcesLive 
            << "' for Streammuxer '" << GetName() << "'");
        m_pStreamMux->SetAttribute("live-source", m_areSourcesLive);
    }

    uint PipelineSourcesBintr::GetStreamMuxNvbufMemType()
    {
        LOG_FUNC();

        return m_nvbufMemType;
    }

    void PipelineSourcesBintr::SetStreamMuxNvbufMemType(uint type)
    {
        LOG_FUNC();

        m_nvbufMemType = type;
        m_pStreamMux->SetAttribute("nvbuf-memory-type", m_nvbufMemType);
    }

    void PipelineSourcesBintr::GetStreamMuxBatchProperties(guint* batchSize, 
        guint* batchTimeout)
    {
        LOG_FUNC();

        *batchSize = m_batchSize;
        *batchTimeout = m_batchTimeout;
    }

    void PipelineSourcesBintr::SetStreamMuxBatchProperties(uint batchSize, 
        uint batchTimeout)
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
        
        LOG_INFO("Setting StreamMux attribute: enable-padding = " 
            << m_isPaddingEnabled); 
        
        m_pStreamMux->SetAttribute("enable-padding", m_isPaddingEnabled);
    }

    void PipelineSourcesBintr::GetStreamMuxNumSurfacesPerFrame(uint* num)
    {
        LOG_FUNC();
        
        m_pStreamMux->GetAttribute("num-surfaces-per-frame", &m_numSurfacesPerFrame);
        *num = m_numSurfacesPerFrame;
    }
    
    void PipelineSourcesBintr::SetStreamMuxNumSurfacesPerFrame(uint num)
    {
        LOG_FUNC();
        
        m_numSurfacesPerFrame = num;
        
        LOG_INFO("Setting StreamMux attribute: num-surfaces-per-frame = " 
            << m_numSurfacesPerFrame); 
        
        m_pStreamMux->SetAttribute("num-surfaces-per-frame", m_numSurfacesPerFrame);
    }
    
    void PipelineSourcesBintr::DisableEosConsumers()
    {
        for (auto const& imap: m_pChildSources)
        {
            imap.second->DisableEosConsumer();
        }
        // If at lease one RTSP Source was added and the EOS Consumer
        // needs to be removed. 
        if (m_pEosConsumer)
        {
            m_pSrcPadProbe->RemovePadProbeHandler(m_pEosConsumer);
        }
    }
}
