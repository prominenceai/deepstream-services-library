/*
The MIT License

Copyright (c) 2019-2025, Prominence AI, Inc.

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
#include "DslStreammuxBintr.h"
#include "DslServices.h"

namespace DSL
{
    StreammuxBintr::StreammuxBintr(const char* name, 
        GstObject* parentBin, uint uniquePipelineId, const char* ghostPadName)
        : Bintr(name, parentBin)
        , m_uniquePipelineId(uniquePipelineId)
        , m_ghostPadName(ghostPadName)
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

        // Need to forward all children messages for this StreammuxBintr,
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
            SetDimensions(DSL_STREAMMUX_DEFAULT_WIDTH, 
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

        // Float the Streammux as a src Ghost Pad for this StreammuxBintr
        m_pStreammux->AddGhostPadToParent("src", ghostPadName);

        // Add the Buffer and DS Event Probes to the Streammuxer - src-pad only.
        AddSrcPadProbes(m_pStreammux->GetGstElement());
        
        // If the unqiue pipeline-id is greater than 0, then we need to add the
        // SourceIdOffsetterPadProbeHandler to offset every source-id found in
        // the frame-metadata produced by the streammux plugin. 
        if (m_uniquePipelineId > 0)
        {
            LOG_INFO("Adding source-id-offsetter to StreammuxBintr '"
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

    StreammuxBintr::~StreammuxBintr()
    {
        LOG_FUNC();

        m_pStreammux->RemoveGhostPadFromParent(m_ghostPadName.c_str());

        // If the unqiue pipeline-id is greater than 0, then we need to remove the
        // SourceIdOffsetterPadProbeHandler
        if (m_uniquePipelineId > 0)
        {
            m_pSrcPadBufferProbe->RemovePadProbeHandler(m_pSourceIdOffsetter);
            m_pSourceIdOffsetter = nullptr;
        }
    }

    bool StreammuxBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("StreammuxBintr '" << GetName() 
                << "' is already linked");
            return false;
        }
        // single element, nothing to link
        m_isLinked = true;
        return true;
    }        

    void StreammuxBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("StreammuxBintr '" << GetName() << "' is not linked");
            return;
        }
        // single element, nothing to link
        m_isLinked = false;
    }

    bool StreammuxBintr::PlayTypeIsLiveGet()
    {
        LOG_FUNC();
        
        return m_areSourcesLive;
    }
    
    bool StreammuxBintr::PlayTypeIsLiveSet(bool isLive)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't update live-source property for StreammuxBintr '" 
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

    const char* StreammuxBintr::GetConfigFile()
    {
        return m_streammuxConfigFile.c_str();
    }
    
    bool StreammuxBintr::SetConfigFile(const char* configFile)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update config-file for StreammuxBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_streammuxConfigFile = configFile;
        m_pStreammux->SetAttribute("config-file-path", 
            m_streammuxConfigFile.c_str());
        
        return true;
    }
    
    uint StreammuxBintr::GetBatchSize()
    {
        LOG_FUNC();

        return m_batchSize;
    }

    bool StreammuxBintr::SetBatchSize(uint batchSize)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update batch-size for StreammuxBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        // Important! once set, this flag cannot be unset.
        m_batchSizeSetByClient = true;
        m_batchSize = batchSize;
        m_pStreammux->SetAttribute("batch-size", m_batchSize);
        
        return true;
    }
    
    uint StreammuxBintr::GetNumSurfacesPerFrame()
    {
        LOG_FUNC();
        
        m_pStreammux->GetAttribute("num-surfaces-per-frame", &m_numSurfacesPerFrame);
        return m_numSurfacesPerFrame;
    }
    
    bool StreammuxBintr::SetNumSurfacesPerFrame(uint num)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR(
                "Can't update num-surfaces-per-frame for StreammuxBintr '"
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_numSurfacesPerFrame = num;
        m_pStreammux->SetAttribute("num-surfaces-per-frame", m_numSurfacesPerFrame);
        
        return true;
    }

    boolean StreammuxBintr::GetSyncInputsEnabled()
    {
        LOG_FUNC();
        
        m_pStreammux->GetAttribute("sync-inputs", &m_syncInputs);
        return m_syncInputs;
    }
    
    bool StreammuxBintr::SetSyncInputsEnabled(boolean enabled)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't update sync-input for StreammuxBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_syncInputs = enabled;
        m_pStreammux->SetAttribute("sync-inputs", m_syncInputs);
        
        return true;
    }

    boolean StreammuxBintr::GetAttachSysTsEnabled()
    {
        LOG_FUNC();
        
        m_pStreammux->GetAttribute("attach-sys-ts", &m_attachSysTs);
        return m_attachSysTs;
    }
    
    bool StreammuxBintr::SetAttachSysTsEnabled(boolean enabled)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't update sync-input for StreammuxBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_attachSysTs = enabled;
        m_pStreammux->SetAttribute("attach-sys-ts", m_attachSysTs);
        
        return true;
    }

    boolean StreammuxBintr::GetMaxLatency()
    {
        LOG_FUNC();
        
        m_pStreammux->GetAttribute("max-latency", &m_maxLatency);
        return m_maxLatency;
    }
    
    bool StreammuxBintr::SetMaxLatency(uint maxLatency)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't update max-latency property for StreammuxBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_maxLatency = maxLatency;
        m_pStreammux->SetAttribute("max-latency", m_maxLatency);
        
        return true;
    }

    void StreammuxBintr::GetBatchProperties(uint* batchSize, 
        int* batchTimeout)
    {
        LOG_FUNC();

        *batchSize = m_batchSize;
        *batchTimeout = m_batchTimeout;
    }

    bool StreammuxBintr::SetBatchProperties(uint batchSize, 
        int batchTimeout)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update batch properties for StreammuxBintr '" 
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
    
    uint StreammuxBintr::GetNvbufMemType()
    {
        LOG_FUNC();

        return m_nvbufMemType;
    }

    bool StreammuxBintr::SetNvbufMemType(uint type)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update nvbuf-memory-type for StreammuxBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_nvbufMemType = type;
        m_pStreammux->SetAttribute("nvbuf-memory-type", m_nvbufMemType);
        
        return true;
    }

    bool StreammuxBintr::SetGpuId(uint gpuId)
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
        
        LOG_INFO("StreammuxBintr '" << GetName() 
            << "' - new GPU ID = " << m_gpuId );
            
        return true;
    }

    void StreammuxBintr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        *width = m_streamMuxWidth;
        *height = m_streamMuxHeight;
    }

    bool StreammuxBintr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update Streammux dimensions for StreammuxBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_streamMuxWidth = width;
        m_streamMuxHeight = height;

        m_pStreammux->SetAttribute("width", m_streamMuxWidth);
        m_pStreammux->SetAttribute("height", m_streamMuxHeight);
        
        return true;
    }
    
    boolean StreammuxBintr::GetPaddingEnabled()
    {
        LOG_FUNC();
        
        return m_isPaddingEnabled;
    }
    
    bool StreammuxBintr::SetPaddingEnabled(boolean enabled)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Can't update enable-padding property for StreammuxBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_isPaddingEnabled = enabled;
        
        m_pStreammux->SetAttribute("enable-padding", m_isPaddingEnabled);
        return true;
    }

    void StreammuxBintr::AddEosConsumer()
    {
        LOG_FUNC();

        std::string eventHandlerName = GetName() + "-eos-consumer";
        m_pEosConsumer = DSL_PPEH_EOS_CONSUMER_NEW(eventHandlerName.c_str());
        m_pSrcPadDsEventProbe->AddPadProbeHandler(m_pEosConsumer);
    }

    void StreammuxBintr::RemoveEosConsumer()
    {
        LOG_FUNC();

        if (m_pEosConsumer)
        {
            m_pSrcPadDsEventProbe->RemovePadProbeHandler(m_pEosConsumer);

            // Destroy the EOS Consumer PPH
            m_pEosConsumer = nullptr;
        }
    }
}
