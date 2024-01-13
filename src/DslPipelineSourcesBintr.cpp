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
    PipelineSourcesBintr::PipelineSourcesBintr(const char* name,
        uint uniquePipelineId)
        : Bintr(name)
        , m_uniquePipelineId(uniquePipelineId)
        , m_areSourcesLive(false)
        , m_batchSizeSetByClient(false)
        , m_frameDuration(-1)   // workaround for nvidia bug
        , m_useNewStreammux(false)
    {
        LOG_FUNC();

        const char* value = getenv("USE_NEW_NVSTREAMMUX");
        if (value and std::string(value) == "yes")
        {
            LOG_WARN(
                "USE_NEW_NVSTREAMMUX is set to yes - enabling new Streammux Services");
            m_useNewStreammux = true;
        }

        // Need to forward all children messages for this PipelineSourcesBintr,
        // which is the parent bin for the Pipeline's Streammux, so the Pipeline
        // can be notified of individual source EOS events. 
        g_object_set(m_pGstObj, "message-forward", TRUE, NULL);
  
        // Single Stream Muxer element for all Sources 
        m_pStreammux = DSL_ELEMENT_NEW("nvstreammux", name);
        
        // Get property defaults that aren't specifically set
        m_pStreammux->GetAttribute("num-surfaces-per-frame", &m_numSurfacesPerFrame);
        m_pStreammux->GetAttribute("attach-sys-ts", &m_attachSysTs);
        m_pStreammux->GetAttribute("sync-inputs", &m_syncInputs);
        m_pStreammux->GetAttribute("max-latency", &m_maxLatency);
        m_pStreammux->GetAttribute("drop-pipeline-eos", &m_dropPipelineEos);

        // IMPORTANT! NVIDIA bug - always returns 18446744073709.
//        m_pStreammux->GetAttribute("frame-duration", &frameDuration);
        m_frameDuration = GST_CLOCK_TIME_NONE;
        
        LOG_INFO("");
        LOG_INFO("Initial property values for Streammux '" << name << "'");
        LOG_INFO("  num-surfaces-per-frame : " << m_numSurfacesPerFrame);
        LOG_INFO("  attach-sys-ts          : " << m_attachSysTs);
        LOG_INFO("  sync-inputs            : " << m_syncInputs);
        LOG_INFO("  max-latency            : " << m_maxLatency);
        LOG_INFO("  frame-duration         : " << m_frameDuration);
        LOG_INFO("  drop-pipeline-eos      : " << m_dropPipelineEos);

        if (!m_useNewStreammux)
        {
            // Must update the default dimensions of 0x0 or the Pipeline
            // will fail to play;
            SetStreammuxDimensions(DSL_STREAMMUX_DEFAULT_WIDTH, 
                DSL_STREAMMUX_DEFAULT_HEIGHT);
                
            m_pStreammux->GetAttribute("batched-push-timeout", &m_batchTimeout);
            m_pStreammux->GetAttribute("enable-padding", &m_isPaddingEnabled);
            m_pStreammux->GetAttribute("gpu-id", &m_gpuId);
            m_pStreammux->GetAttribute("nvbuf-memory-type", &m_nvbufMemType);
            m_pStreammux->GetAttribute("buffer-pool-size", &m_bufferPoolSize);
            
            LOG_INFO("  width                  : " << m_streamMuxWidth);
            LOG_INFO("  height                 : " << m_streamMuxHeight);
            LOG_INFO("  batched-push-timeout   : " << m_batchTimeout);
            LOG_INFO("  enable-padding         : " << m_isPaddingEnabled);
            LOG_INFO("  gpu-id                 : " << m_gpuId);
            LOG_INFO("  nvbuf-memory-type      : " << m_nvbufMemType);
            LOG_INFO("  buffer-pool-size       : " << m_bufferPoolSize);
        }

        AddChild(m_pStreammux);

        // Float the Streammux as a src Ghost Pad for this PipelineSourcesBintr
        m_pStreammux->AddGhostPadToParent("src");

        // Add the Buffer and DS Event Probes to the Streammuxer - src-pad only.
        AddSrcPadProbes(m_pStreammux);
        
        // If the unqiue pipeline-id is greater than 0, then we need to add the
        // SourceIdOffsetterPadProbeHandler to offset every source-id found in
        // the frame-metadata produced by the streammux plugin. 
        if (m_uniquePipelineId > 0)
        {
            LOG_INFO("Adding source-id-offsetter to PipelineSourcesBintr '"
                << GetName() << "' with unique Pipeline-id = " << m_uniquePipelineId);

            // Create the specialized pad-probe-handler to offset all source-ids'
            std::string bufferHandlerName = GetName() + "-source-id-offsetter";
            m_pSourceIdOffsetter = DSL_PPH_SOURCE_ID_OFFSETTER_NEW(
                bufferHandlerName.c_str(), 
                (m_uniquePipelineId << DSL_PIPELINE_SOURCE_UNIQUE_ID_OFFSET_IN_BITS));

            // Add the specialized handler to the buffer-pad-probe. 
            m_pSrcPadBufferProbe->AddPadProbeHandler(m_pSourceIdOffsetter);
        }
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
            StreammuxPlayTypeIsLiveSet(pChildSource->IsLive());
        }
        else if (pChildSource->IsLive() != StreammuxPlayTypeIsLiveGet())
        {
            LOG_ERROR("Can't add Source '" << pChildSource->GetName() 
                << "' with IsLive=" << pChildSource->IsLive()  << " to streamuxer '" 
                << GetName() << "' with IsLive=" << StreammuxPlayTypeIsLiveGet());
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
                
                // Create the Pad Probe and EOS Consumer to drop the EOS event that 
                // occurs on loss of RTSP stream, allowing the Pipeline to continue 
                // to play. Each RTSP source will then manage their own restart 
                // attempts and time management.

                std::string eventHandlerName = GetName() + "-eos-consumer";
                m_pEosConsumer = DSL_PPEH_EOS_CONSUMER_NEW(eventHandlerName.c_str());
                m_pSrcPadDsEventProbe->AddPadProbeHandler(m_pEosConsumer);
            }
        }
        
        uint padId(0);
        
        // find the next available unused stream-id
        auto ivec = find(m_usedRequestPadIds.begin(), 
            m_usedRequestPadIds.end(), false);
        
        // If we're inserting into the location of a previously remved source
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
        // Set the source's request sink pad-id
        pChildSource->SetRequestPadId(padId);

        // Set the sources unique id by shifting/or-ing the unique pipeline-id
        // with the source's pad-id -- combined, they are gauranteed to be unique.
        pChildSource->SetUniqueId(
            (m_uniquePipelineId << DSL_PIPELINE_SOURCE_UNIQUE_ID_OFFSET_IN_BITS) 
            | padId);
            
        // Add the source's "unique-name to unique-id" mapping to the Services DB.
        Services::GetServices()->_sourceNameSet(pChildSource->GetCStrName(),
            pChildSource->GetUniqueId());

        // Add the Source to the Bintrs collection of children mapped by name
        m_pChildSources[pChildSource->GetName()] = pChildSource;
        
        // Add the Source to the Bintrs collection of children mapped by padId
        m_pChildSourcesIndexed[padId] = pChildSource;
        
        // call the parent class to complete the add
        if (!Bintr::AddChild(pChildSource))
        {
            LOG_ERROR("Faild to add Source '" << pChildSource->GetName() 
                << "' as a child to '" << GetName() << "'");
            return false;
        }
        
        // If the Pipeline is currently in a linked state, Set child source 
        // Id to the next available, linkAll Elementrs now and Link to the 
        // Stream-muxer
        if (IsLinked())
        {
            std::string sinkPadName = "sink_" + std::to_string(padId);
            
            if (!pChildSource->LinkAll() or 
                !pChildSource->LinkToSinkMuxer(m_pStreammux, sinkPadName.c_str()))
            {
                LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                    << "' failed to Link Child Source '" 
                    << pChildSource->GetName() << "'");
                return false;
            }
            if (!m_batchSizeSetByClient)
            {
                // Increment the current batch-size
                m_batchSize++;
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
            LOG_INFO("Unlinking " << m_pStreammux->GetName() << " from " 
                << pChildSource->GetName());
                
            // unlink the source from the Streammuxer
            if (!pChildSource->UnlinkFromSinkMuxer())
            {
                LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                    << "' failed to Unlink Child Source '" 
                    << pChildSource->GetName() << "'");
                return false;
            }
            // unlink all of the ChildSource's Elementrs
            pChildSource->UnlinkAll();

            if (!m_batchSizeSetByClient)
            {
                // Decrement the current batch-size
                m_batchSize--;
            }
        }
        // Erase the Source the source from the name<->unique-id database.
        Services::GetServices()->_sourceNameErase(pChildSource->GetCStrName());
        
        // unreference and remove from the child source collections
        m_pChildSources.erase(pChildSource->GetName());
        m_pChildSourcesIndexed.erase(pChildSource->GetRequestPadId());

        // set the used-stream id as available for reuse
        m_usedRequestPadIds[pChildSource->GetRequestPadId()] = false;
        pChildSource->SetRequestPadId(-1);
        pChildSource->SetUniqueId(-1);
        
        // call the base function to complete the remove
        return Bintr::RemoveChild(pChildSource);
    }
    
    bool PipelineSourcesBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                << "' is already linked");
            return false;
        }
        
        for (auto const& imap: m_pChildSourcesIndexed)
        {
            std::string sinkPadName = 
                "sink_" + std::to_string(imap.second->GetRequestPadId());
            
            if (!imap.second->LinkAll() or 
                !imap.second->LinkToSinkMuxer(m_pStreammux,
                    sinkPadName.c_str()))
            {
                LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                    << "' failed to Link Child Source '" 
                    << imap.second->GetName() << "'");
                return false;
            }
        }
        // Set the Batch size to the nuber of sources owned if not already set
        if (!m_batchSizeSetByClient)
        {
            m_pStreammux->SetAttribute("batch-size", m_batchSize);
            m_batchSize = m_pChildSources.size();
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
            LOG_INFO("Unlinking " << m_pStreammux->GetName() 
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
        // Set the Batch size to the nuber of sources owned if not already set
        if (!m_batchSizeSetByClient)
        {
            m_batchSize = 0;
        }
        m_isLinked = false;
    }
    
    void PipelineSourcesBintr::EosAll()
    {
        LOG_FUNC();
        
        // Send EOS message to each source object.
        for (auto const& imap: m_pChildSources)
        {
            LOG_INFO("Sending EOS message to Source "  << imap.second->GetName());
            gst_element_send_event(imap.second->GetGstElement(), 
                gst_event_new_eos());
        }
    }
    
    
    bool PipelineSourcesBintr::StreammuxPlayTypeIsLiveGet()
    {
        LOG_FUNC();
        
        return m_areSourcesLive;
    }
    
    bool PipelineSourcesBintr::StreammuxPlayTypeIsLiveSet(bool isLive)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't update live-source property for PipelineSourcesBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_areSourcesLive = isLive;
        
        if (!m_useNewStreammux)
        {
            LOG_INFO("'live-source' attrubute set to '" << m_areSourcesLive 
                << "' for Streammuxer '" << GetName() << "'");
            
            m_pStreammux->SetAttribute("live-source", m_areSourcesLive);
        }
        return true;
    }

    const char* PipelineSourcesBintr::GetStreammuxConfigFile()
    {
        return m_streammuxConfigFile.c_str();
    }
    
    bool PipelineSourcesBintr::SetStreammuxConfigFile(const char* configFile)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update config-file for PipelineSourcesBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_streammuxConfigFile = configFile;
        m_pStreammux->SetAttribute("config-file-path", 
            m_streammuxConfigFile.c_str());
        
        return true;
    }
    
    uint PipelineSourcesBintr::GetStreammuxBatchSize()
    {
        LOG_FUNC();

        return m_batchSize;
    }

    bool PipelineSourcesBintr::SetStreammuxBatchSize(uint batchSize)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update batch-size for PipelineSourcesBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        // Important! once set, this flag cannot be unset.
        m_batchSizeSetByClient = true;
        m_batchSize = batchSize;
        m_pStreammux->SetAttribute("batch-size", m_batchSize);
        
        return true;
    }
    
    uint PipelineSourcesBintr::GetStreammuxNumSurfacesPerFrame()
    {
        LOG_FUNC();
        
        m_pStreammux->GetAttribute("num-surfaces-per-frame", &m_numSurfacesPerFrame);
        return m_numSurfacesPerFrame;
    }
    
    bool PipelineSourcesBintr::SetStreammuxNumSurfacesPerFrame(uint num)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR(
                "Can't update num-surfaces-per-frame for PipelineSourcesBintr '"
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_numSurfacesPerFrame = num;
        m_pStreammux->SetAttribute("num-surfaces-per-frame", m_numSurfacesPerFrame);
        
        return true;
    }

    boolean PipelineSourcesBintr::GetStreammuxSyncInputsEnabled()
    {
        LOG_FUNC();
        
        m_pStreammux->GetAttribute("sync-inputs", &m_syncInputs);
        return m_syncInputs;
    }
    
    bool PipelineSourcesBintr::SetStreammuxSyncInputsEnabled(boolean enabled)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't update sync-input for PipelineSourcesBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_syncInputs = enabled;
        m_pStreammux->SetAttribute("sync-inputs", m_syncInputs);
        
        return true;
    }

    boolean PipelineSourcesBintr::GetStreammuxAttachSysTsEnabled()
    {
        LOG_FUNC();
        
        m_pStreammux->GetAttribute("attach-sys-ts", &m_attachSysTs);
        return m_attachSysTs;
    }
    
    bool PipelineSourcesBintr::SetStreammuxAttachSysTsEnabled(boolean enabled)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't update sync-input for PipelineSourcesBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_attachSysTs = enabled;
        m_pStreammux->SetAttribute("attach-sys-ts", m_attachSysTs);
        
        return true;
    }

    boolean PipelineSourcesBintr::GetStreammuxMaxLatency()
    {
        LOG_FUNC();
        
        m_pStreammux->GetAttribute("max-latency", &m_maxLatency);
        return m_maxLatency;
    }
    
    bool PipelineSourcesBintr::SetStreammuxMaxLatency(uint maxLatency)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't update max-latency property for PipelineSourcesBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_maxLatency = maxLatency;
        m_pStreammux->SetAttribute("max-latency", m_maxLatency);
        
        return true;
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
            m_pSrcPadDsEventProbe->RemovePadProbeHandler(m_pEosConsumer);
        }
    }

    void PipelineSourcesBintr::GetStreammuxBatchProperties(uint* batchSize, 
        int* batchTimeout)
    {
        LOG_FUNC();

        *batchSize = m_batchSize;
        *batchTimeout = m_batchTimeout;
    }

    bool PipelineSourcesBintr::SetStreammuxBatchProperties(uint batchSize, 
        int batchTimeout)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update batch properties for PipelineSourcesBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_batchSizeSetByClient = true;
        m_batchSize = batchSize;
        m_batchTimeout = batchTimeout;

        m_pStreammux->SetAttribute("batch-size", m_batchSize);
        m_pStreammux->SetAttribute("batched-push-timeout", m_batchTimeout);
        
        return true;
    }
    
    uint PipelineSourcesBintr::GetStreammuxNvbufMemType()
    {
        LOG_FUNC();

        return m_nvbufMemType;
    }

    bool PipelineSourcesBintr::SetStreammuxNvbufMemType(uint type)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update nvbuf-memory-type for PipelineSourcesBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_nvbufMemType = type;
        m_pStreammux->SetAttribute("nvbuf-memory-type", m_nvbufMemType);
        
        return true;
    }

    bool PipelineSourcesBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set GPU ID for Pipeline '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        m_gpuId = gpuId;
        m_pStreammux->SetAttribute("gpu-id", m_gpuId);
        
        LOG_INFO("PipelineSourcesBintr '" << GetName() 
            << "' - new GPU ID = " << m_gpuId );
            
        return true;
    }

    void PipelineSourcesBintr::GetStreammuxDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_streamMuxWidth;
        *height = m_streamMuxHeight;
    }

    bool PipelineSourcesBintr::SetStreammuxDimensions(uint width, uint height)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update Streammux dimensions for PipelineSourcesBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_streamMuxWidth = width;
        m_streamMuxHeight = height;

        m_pStreammux->SetAttribute("width", m_streamMuxWidth);
        m_pStreammux->SetAttribute("height", m_streamMuxHeight);
        
        return true;
    }
    
    boolean PipelineSourcesBintr::GetStreammuxPaddingEnabled()
    {
        LOG_FUNC();
        
        return m_isPaddingEnabled;
    }
    
    bool PipelineSourcesBintr::SetStreammuxPaddingEnabled(boolean enabled)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't update enable-padding property for PipelineSourcesBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_isPaddingEnabled = enabled;
        
        m_pStreammux->SetAttribute("enable-padding", m_isPaddingEnabled);
        return true;
    }

}
