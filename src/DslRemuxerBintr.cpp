/*
The MIT License

Copyright (c) 2023-2024, Prominence AI, Inc.

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
    
    //--------------------------------------------------------------------------------

    RemuxerBranchBintr::RemuxerBranchBintr(const char* name, 
        GstObject* parentRemuxerBin, 
        DSL_BINTR_PTR pChildBranch, uint* streamIds, uint numStreamIds)
        : Bintr(name, parentRemuxerBin)
        , m_pChildBranch(pChildBranch)
        , m_linkSelectiveStreams(numStreamIds)   // true if numStreamIds > 0
        , m_frameDuration(GST_CLOCK_TIME_NONE)   // workaround for nvidia bug
        , m_useNewStreammux(false)
        , m_width(0)
        , m_height(0)
    {
        LOG_FUNC();

        const char* value = getenv("USE_NEW_NVSTREAMMUX");
        if (value and std::string(value) == "yes")
        {
            LOG_WARN(
                "USE_NEW_NVSTREAMMUX is set to yes - enabling new Streammux Services");
            m_useNewStreammux = true;
        }

        // Need to forward all children messages for this RemuxerBranchBintr,
        g_object_set(m_pGstObj, "message-forward", TRUE, NULL);

        // If linking to specific streams ids - not all.
        if (streamIds and numStreamIds)
        {
            // Setup the branches' vector of seclect stream-id
            m_streamIds.assign(streamIds, streamIds+numStreamIds);

            // Build the config string for this RemuxerBranch Bintr using
            // format = <prefix>-<model unique ID>=<source ids>.
            std::stringstream ssStreamIds;
            ssStreamIds << DSL_REMUXER_BRANCH_CONFIG_STRING_PREFIX 
                << std::to_string(pChildBranch->GetUniqueId()) << "=";
            std::copy(m_streamIds.begin(), m_streamIds.end()-1,
                std::ostream_iterator<uint>(ssStreamIds, ";"));
            ssStreamIds << m_streamIds.back();
            m_branchConfigString = ssStreamIds.str();
        }

        // Create a new Streammuxer for this child branch with a unique name
        // derived from this RemuxerBranchBintr's name which comes from the
        // unique pChildBranch name - enforced by the client API.
        std::string streammuxerName = GetName() + "-streammuxer";
        
        // Each branch needs its own Streammuxer to connect back to the output Tees,
        // which are connected to the request-pads of the Demuxer.
        m_pStreammux = DSL_ELEMENT_NEW("nvstreammux", streammuxerName.c_str());

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
        LOG_INFO("Initial property values for RemuxerBranchBintr '" << name << "'");
        LOG_INFO("  child-branch           : " << m_pChildBranch->GetName());
        LOG_INFO("  config-string          : " << m_branchConfigString);
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
            m_pStreammux->GetAttribute("gpu-id", &m_gpuId);
            m_pStreammux->GetAttribute("nvbuf-memory-type", &m_nvbufMemType);
            
            LOG_INFO("  width                  : " << m_width);
            LOG_INFO("  height                 : " << m_height);
            LOG_INFO("  batched-push-timeout   : " << m_batchTimeout);
            LOG_INFO("  gpu-id                 : " << m_gpuId);
            LOG_INFO("  nvbuf-memory-type      : " << m_nvbufMemType);
        }

       
        // Add both the ChildBranch and new Streammuxer to this RemuxerBranchBintr
        // so that we can link them together => streammuxer->branch.
        AddChild(m_pChildBranch);
        AddChild(m_pStreammux);
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
        if (!m_batchSize)
        {
            LOG_ERROR("Can't link RemuxerBranchBintr '" << GetName() 
                << "' as batch-size is not set");
            return false;
        }
        // Need to determine if the Branch is to be connected to a 
        // select set of stream-ids or all in batch-size
        if(!m_linkSelectiveStreams)
        {
            LOG_INFO("Connecting to all streams for RemuxerBranchBintr '"
                << GetName() << "'");
            for (auto i=0; i<m_batchSize; i++)
            {
                m_streamIds.push_back(i);
            }
        }
        // We need to create Queue elements to link to the request src pads
        // for each Tee so that each stream can be linked as follows.
        //
        // Demuxer[stream-i]->Tee[pad-id]->Queue->[stream-i]Streammuxer
        //
        // The first Branch added will link to each Tee on request-pad-id = 0.
        // The second Branch added will link to each Tee on request-pad-d = 1.
        for (auto i=0; i<m_streamIds.size(); i++)
        {
            DSL_ELEMENT_PTR pQueue = DSL_ELEMENT_EXT_NEW("queue", 
                GetCStrName(), std::to_string(m_streamIds[i]).c_str());

            // Add the new queue as a child of this proxy bintr - proxy for the 
            // RemuxerBintr.  Add the Queue to the map of Queues to be linked
            // to the Remuxer Tees in LinkToSourceTees() below
            AddChild(pQueue);
            m_queues[m_streamIds[i]] = pQueue;

            std::string sinkPadName = 
                "sink_" + std::to_string(m_streamIds[i]);
            
            if (!pQueue->LinkToSinkMuxer(m_pStreammux, "src", sinkPadName.c_str()))
            {
                return false;
            }
        }
        
        // Propagate the link method and batch size to the Child Branch
        m_pChildBranch->SetLinkMethod(m_linkMethod);
        m_pChildBranch->SetBatchSize(m_batchSize);
            
        // Then link the Streammuxer to the Child Branch
        if (!m_pChildBranch->LinkAll() or
            !m_pStreammux->LinkToSink(m_pChildBranch))
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
        for (auto i=0; i<m_streamIds.size(); i++)
        {
            // Unlink the Queue for the Streammuxer and remove as Child
            // of the Proxy - Bintr for the RemuxerBintr
            m_queues[m_streamIds[i]]->UnlinkFromSinkMuxer(m_pStreammux, "src");

            RemoveChild(m_queues[m_streamIds[i]]);
        }
        // Delete all child Queue elements
        m_queues.clear();
        m_pStreammux->UnlinkFromSink();
        m_pChildBranch->UnlinkAll();
        
        // If we're not linking to specific stream-ids, clear the stream-id 
        // vector to be populated on next LinkAll()
        if (!m_linkSelectiveStreams)
        {
            m_streamIds.clear();
        }

        m_isLinked = false;
    }

    bool RemuxerBranchBintr::LinkToSourceTees(
        const std::vector<DSL_ELEMENT_PTR>& tees)
    {
        LOG_FUNC();
    
        // We loop through the stream-ids vector to link each Queue with
        // the correct Remuxer Tee.
        for (auto i=0; i<m_streamIds.size(); i++)
        {
            if (!m_queues[m_streamIds[i]]->LinkToSourceTee(
                tees[m_streamIds[i]], "src_%u"))
            {
                LOG_INFO("Failed to link queue '" 
                    << m_queues[m_streamIds[i]]->GetName() 
                    << "' back to source-tee '" 
                    << tees[m_streamIds[i]]->GetName() << "'");
                return false;
            }
                
            LOG_INFO("Successfully linked queue '" 
                << m_queues[m_streamIds[i]]->GetName() 
                << "' back to source-tee '" 
                << tees[m_streamIds[i]]->GetName() << "'");
        }
        return true;
    }
    
    void RemuxerBranchBintr::UnlinkFromSourceTees()
    {
        LOG_FUNC();
    
        // We loop through the stream-ids vector to unlink each Queue from
        // the RemuxerBintr's source Tees.
        for (auto i=0; i<m_streamIds.size(); i++)
        {
            if (!m_queues[m_streamIds[i]]->UnlinkFromSourceTee())
            {
                LOG_INFO("Failed to unlink queue '" 
                    << m_queues[m_streamIds[i]]->GetName() 
                    << "' from source-tee");
            } 
            else
            {
                LOG_INFO("Successfully unlinked queue '" 
                    << m_queues[m_streamIds[i]]->GetName() 
                    << "' from source-tee");
            }
        }
    }

    bool RemuxerBranchBintr::LinkToSinkMuxer(DSL_NODETR_PTR pMuxer, 
        const char* padName)
    {
        LOG_FUNC();

        return m_pChildBranch->LinkToSinkMuxer(pMuxer, "src", padName);
    }

    bool RemuxerBranchBintr::UnlinkFromSinkMuxer(DSL_NODETR_PTR pMuxer)
    {
        LOG_FUNC();
        
        return m_pChildBranch->UnlinkFromSinkMuxer(pMuxer, "src");
    }
    
    uint RemuxerBranchBintr::GetBatchSize()
    {
        LOG_FUNC();

        return m_batchSize;
    }

    bool RemuxerBranchBintr::SetBatchSize(uint batchSize)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update batch properties for RemuxerBranchBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        // If the Branch is linking to select streams
        if (m_linkSelectiveStreams)
        {
            // Make sure the number of target stream-ids is no greater than batch-size
            if (m_streamIds.size() > batchSize)
            {
                LOG_ERROR("num-stream-ids '" << m_streamIds.size() 
                    << "' is greater than batch-size '" << batchSize
                    << "' for RemuxerBranchBintr '" << GetName() << "'" );
                return false;
            }
            // Make sure that each stream-id is within the batch-size.
            for (const auto& ivec: m_streamIds)
            {
                if (ivec >= batchSize)
                {
                    LOG_ERROR("Stream-id '" << ivec 
                        << "' is >= batch-size '" << batchSize
                        << "' for RemuxerBranchBintr '" << GetName() << "'" );
                    return false;
                }
            }
            // If linking to specific streams, use the size of the stream-id vector
            // for batch-size
            m_batchSize = m_streamIds.size();
        }
        else
        {
            // otherwise, use parent Remuxer batch-size
            m_batchSize = batchSize;
        }
        
        m_pStreammux->SetAttribute("batch-size", m_batchSize);
        
        return true;
    }

    const char* RemuxerBranchBintr::GetStreammuxConfigFile()
    {
        return m_streammuxConfigFile.c_str();
    }
    
    bool RemuxerBranchBintr::SetStreammuxConfigFile(const char* configFile)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update config-file for RemuxerBranchBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_streammuxConfigFile = configFile;
        m_pStreammux->SetAttribute("config-file-path", 
            m_streammuxConfigFile.c_str());
        
        return true;
    }

    void RemuxerBranchBintr::GetBatchProperties(uint* batchSize, 
        int* batchTimeout)
    {
        LOG_FUNC();

        *batchSize = m_batchSize;
        *batchTimeout = m_batchTimeout;
    }

    bool RemuxerBranchBintr::SetBatchProperties(uint batchSize, 
        int batchTimeout)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update batch properties for RemuxerBranchBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        // If the Branch is linking to select streams
        if (m_linkSelectiveStreams)
        {
            // Make sure the number of target stream-ids is no greater than batch-size
            if (m_streamIds.size() > batchSize)
            {
                LOG_ERROR("num-stream-ids '" << m_streamIds.size() 
                    << "' is greater than batch-size '" << batchSize
                    << "' for RemuxerBranchBintr '" << GetName() << "'" );
                return false;
            }
            // Make sure that each stream-id is within the batch-size.
            for (const auto& ivec: m_streamIds)
            {
                if (ivec >= batchSize)
                {
                    LOG_ERROR("Stream-id '" << ivec 
                        << "' is >= batch-size '" << batchSize
                        << "' for RemuxerBranchBintr '" << GetName() << "'" );
                    return false;
                }
            }
            // If linking to specific streams, use the size of the stream-id vector
            // for batch-size
            m_batchSize = m_streamIds.size();
        }
        else
        {
            // otherwise, use parent Remuxer batch-size
            m_batchSize = batchSize;
        }
        m_batchTimeout = batchTimeout;

        m_pStreammux->SetAttribute("batch-size", m_batchSize);
        m_pStreammux->SetAttribute("batched-push-timeout", m_batchTimeout);

        return true;
    }

    void RemuxerBranchBintr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();

        *width = m_width;
        *height = m_height;
    }

    bool RemuxerBranchBintr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update dimensions for RemuxerBranchBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_width = width;
        m_height = height;

        m_pStreammux->SetAttribute("width", m_width);
        m_pStreammux->SetAttribute("height", m_height);
        return true;
    }

    bool RemuxerBranchBintr::SetNvbufMemType(uint type)
    {
        LOG_FUNC();

        if (m_useNewStreammux)
        {
            LOG_ERROR("Can't update nvbuf-memory-type for RemuxerBranchBintr '" 
                << GetName() << "' USE_NEW_NVSTREAMMUX must NOT be set to 'yes'");
            return false;
        }
        if (m_isLinked)
        {
            LOG_ERROR("Can't update nvbuf-memory-type for RemuxerBranchBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        if (m_nvbufMemType != type)
        {
            m_nvbufMemType = type;
            m_pStreammux->SetAttribute("nvbuf-memory-type", m_nvbufMemType);
        }
        return true;
    }

    bool RemuxerBranchBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (m_useNewStreammux)
        {
            LOG_ERROR("Can't update gpu-id for RemuxerBranchBintr '" 
                << GetName() << "' USE_NEW_NVSTREAMMUX must NOT be set to 'yes'");
            return false;
        }
        if (IsLinked())
        {
            LOG_ERROR("Unable to set GPU ID for Pipeline '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        if (m_gpuId != gpuId)
        {
            m_gpuId = gpuId;
            m_pStreammux->SetAttribute("gpu-id", m_gpuId);
            
            LOG_INFO("PipelineSourcesBintr '" << GetName() 
                << "' - new GPU ID = " << m_gpuId );
        }
        return true;
    }

    //--------------------------------------------------------------------------------
            
    RemuxerBintr::RemuxerBintr(const char* name)
        : QBintr(name)
        , m_batchSizeSetByClient(false)
        , m_useNewStreammux(false)
        , m_batchTimeout(-1)
        , m_width(0)
        , m_height(0)
       // m_batchSize initialized to 0 in Bintr ctor
    {
        LOG_FUNC();

        const char* value = getenv("USE_NEW_NVSTREAMMUX");
        if (value and std::string(value) == "yes")
        {
            LOG_WARN(
                "USE_NEW_NVSTREAMMUX is set to yes - enabling new Streammux Services");
            m_useNewStreammux = true;
        }

        // Config file used to define the stream selection for each branch.
        m_configFilePath = "/tmp/" + GetName() + "_config.txt";

        // Need to forward all children messages for this RemuxerBintr,
        // which is the parent bin for the Streammuxer allocated, so the Pipeline
        // can be notified of individual source EOS events. 
        g_object_set(m_pGstObj, "message-forward", TRUE, NULL);

        // IMPORTANT! The QBintr::m_pQueue is used to connect the input tee
        // to the Metamuxer's active sink pad.         
        m_pInputTee = DSL_ELEMENT_EXT_NEW("tee", name, "input");
        m_pMetamuxer = DSL_ELEMENT_NEW("nvdsmetamux", name);
        m_pDemuxerQueue = DSL_ELEMENT_EXT_NEW("queue", name, "nvstreamdemux");
        m_pDemuxer = DSL_ELEMENT_NEW("nvstreamdemux", name);
        
        m_pMetamuxer->GetAttribute("active-pad", &m_activePad);

        LOG_INFO("");
        LOG_INFO("Initial property values for RemuxerBintr '" << name << "'");
        LOG_INFO("  active-pad             : " << m_activePad);
        
        if (m_useNewStreammux)
        {
            LOG_INFO("  none applicable        : ");
        }
        else
        {
            SetDimensions(DSL_STREAMMUX_DEFAULT_WIDTH, 
                DSL_STREAMMUX_DEFAULT_HEIGHT);

            LOG_INFO("  width                  : " << m_width);
            LOG_INFO("  height                 : " << m_height);
            LOG_INFO("  batched-push-timeout   : " << m_batchTimeout);            
        }
        
        // Add the demuxer as child and elevate as sink ghost pad
        Bintr::AddChild(m_pInputTee);
        Bintr::AddChild(m_pMetamuxer);
        Bintr::AddChild(m_pDemuxerQueue);
        Bintr::AddChild(m_pDemuxer);

        m_pInputTee->AddGhostPadToParent("sink");
        m_pMetamuxer->AddGhostPadToParent("src");

        // Add the Buffer and DS Event probes to the input Tee and Metamuxer.
        AddSinkPadProbes(m_pInputTee->GetGstElement());
        AddSrcPadProbes(m_pMetamuxer->GetGstElement());
    }    
    
    RemuxerBintr::~RemuxerBintr()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }
    
    bool RemuxerBintr::AddChild(DSL_BINTR_PTR pChildComponent)
    {
        LOG_FUNC();

        if (IsChild(pChildComponent))
        {
            LOG_ERROR("Component '" << pChildComponent->GetName() 
                << "' is already a child of '" << GetName() << "'");
            return false;
        }
        if (m_isLinked)
        {
            LOG_ERROR("Can't add branch to RemuxerBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        
        // Important - use the child branch components name. 
        // The Branches are created as proxy-bins for this bintr.
        DSL_REMUXER_BRANCH_PTR pRemuxerBranch = 
            DSL_REMUXER_BRANCH_NEW(pChildComponent->GetCStrName(), GetGstObject(), 
                pChildComponent, NULL, 0);
        
        // Add the branch to the Remuxers collection of children branches 
        m_childBranches[pChildComponent->GetName()] = pRemuxerBranch;
        
        return true;
    }
    
    bool RemuxerBintr::AddChildTo(DSL_BINTR_PTR pChildComponent, 
        uint* streamIds, uint numStreamIds)
    {
        LOG_FUNC();

        if (IsChild(pChildComponent))
        {
            LOG_ERROR("Component '" << pChildComponent->GetName() 
                << "' is already a child of '" << GetName() << "'");
            return false;
        }
        if (m_isLinked)
        {
            LOG_ERROR("Can't add branch to RemuxerBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        
        // Important - use the child branch components name. 
        // The Branches are created as proxy-bins for this bintr.
        DSL_REMUXER_BRANCH_PTR pRemuxerBranch = 
            DSL_REMUXER_BRANCH_NEW(pChildComponent->GetCStrName(), GetGstObject(),
                    pChildComponent, streamIds, numStreamIds);
        
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
        if (m_isLinked)
        {
            LOG_ERROR("Can't remove branch from RemuxerBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        // Remove the BranchBintr from the Remuxers collection of children branches.
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
        if (!m_batchSize)
        {
            LOG_ERROR("Can't link RemuxerBintr '" << m_name 
                << "' as batch-size is not set");
            return false;
        }
        
        // Create the Metamuxer config-file utility.
        RemuxerConfigFile configFile(m_configFilePath);
        
        if (!m_pQueue->LinkToSourceTee(m_pInputTee, "src_%u") or
            !m_pQueue->LinkToSinkMuxer(m_pMetamuxer, "src", "sink_0") or
            !m_pDemuxerQueue->LinkToSourceTee(m_pInputTee, "src_%u") or
            !m_pDemuxerQueue->LinkToSink(m_pDemuxer))
        {
            return false;
        }

        // We need to request all the needed source pads while the 
        // nvstreamdemux plugin is in a NULL state. This is a workaround
        // for the fact the the plugin does not support dynamic requests
        for (uint i=0; i<m_batchSize; i++)
        {
            std::string srcPadName = "src_" + std::to_string(i);
                
            GstPad* pRequestedSrcPad = gst_element_get_request_pad(
                m_pDemuxer->GetGstElement(), srcPadName.c_str());
            if (!pRequestedSrcPad)
            {
                
                LOG_ERROR("Failed to get a requested source pad for Demuxer '" 
                    << GetName() << "'");
                return false;
            }
            LOG_INFO("Allocated requested source pad = " << pRequestedSrcPad 
                << " for RemuxerBintr '" << GetName() << "'");
            m_requestedSrcPads.push_back(pRequestedSrcPad);
            
            // Create a new Tee element and add as a child to the Demuxer.
            DSL_ELEMENT_PTR pTee = DSL_ELEMENT_EXT_NEW("tee", 
                GetCStrName(), std::to_string(i).c_str());
            
            Bintr::AddChild(pTee);
            
            // We can now link the new Tee to the Demuxer request src-pad.
            if (!pTee->LinkToSourceTee(m_pDemuxer, pRequestedSrcPad))
            {
                LOG_ERROR("DemuxerBintr '" << GetName() 
                    << "' failed to Tee Component '" 
                        << pTee->GetName() << "'");
                return false;
            }

            // We add the Tee to the vector of Tees for link access.
            m_tees.push_back(pTee);
        }
        
        // For each Branch, we set the Streammuxers properties based on version
        // Then, call on the Branch to link-all of its children and to link back
        // to the Demuxer source Tees according to their select stream-ids.
        
        
        // Pad index to link the child branch to the Metamuxer
        uint i(1);
        for (auto const& imap: m_childBranches)
        {
            if (UseNewStreammux())
            {
                if (!imap.second->SetBatchSize(m_batchSize))
                {
                    return false;
                }
            }
            else
            {
                if (!imap.second->SetBatchProperties(m_batchSize, m_batchTimeout) or
                    !imap.second->SetNvbufMemType(m_nvbufMemType) or
                    !imap.second->SetGpuId(m_gpuId) or
                    !imap.second->SetDimensions(m_width, m_height))
                {
                    return false;
                }                
            }
            // Propagate the current link-method to each child branch
            imap.second->SetLinkMethod(m_linkMethod);
                
            std::string sinkPadName = "sink_" + std::to_string(i++);
            
            // Linkup each child branch
            if (!imap.second->LinkAll() or
                !imap.second->LinkToSourceTees(m_tees) or
                !imap.second->LinkToSinkMuxer(m_pMetamuxer, sinkPadName.c_str()))
            {
                return false;
            }                
            
            // Get the config string from the child and add it to the config file
            if (imap.second->GetBranchConfigString().size())
            {
                configFile.AddBranchConfigString(
                    imap.second->GetBranchConfigString());
            }
        }
        configFile.Close();
        
        // Pass the complete config file to the Metamuxer
        m_pMetamuxer->SetAttribute("config-file", m_configFilePath.c_str());

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

        m_pQueue->UnlinkFromSourceTee();
        m_pQueue->UnlinkFromSinkMuxer(m_pMetamuxer, "src");
        m_pDemuxerQueue->UnlinkFromSourceTee();
        m_pDemuxerQueue->UnlinkFromSink();

        for (auto const& imap: m_childBranches)
        {
            // Important to unlink from source Tees first, UnlinkAll will 
            // delete all queues.
            imap.second->UnlinkFromSourceTees();
            imap.second->UnlinkFromSinkMuxer(m_pMetamuxer);
            imap.second->UnlinkAll();
        }

        // Now unlink the Tees from the Demuxer, remove as child, and delete.
        while (m_tees.size())
        {
            if (!m_tees.back()->UnlinkFromSourceTee())
            {
                LOG_ERROR("RemuxerBintr '" << GetName() 
                    << "' failed to Unlink Tee '" 
                    << m_tees.back()->GetName() << "'");
                return;
            }
            Bintr::RemoveChild(m_tees.back());
            m_tees.pop_back();
        }
        
        // We now free all of the pre-allocated requested pads
        while (m_requestedSrcPads.size())
        {
            LOG_INFO("Releasing requested source pad = " 
                << m_requestedSrcPads.back()
                << " for DemuxerBintr '"<< GetName() << "'");
                
            gst_element_release_request_pad(m_pDemuxer->GetGstElement(), 
                m_requestedSrcPads.back());
            m_requestedSrcPads.pop_back();
        }
        m_isLinked = false;
    }

    bool RemuxerBintr::AddToParent(DSL_BASE_PTR pBranchBintr)
    {
        LOG_FUNC();
        
        // add 'this' Remuxer to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pBranchBintr)->
            AddRemuxerBintr(shared_from_this());
        return true;
    }

    bool RemuxerBintr::RemoveFromParent(DSL_BASE_PTR pBranchBintr)
    {
        LOG_FUNC();
        
        // remove 'this' Remuxer to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pBranchBintr)->
            RemoveRemuxerBintr(shared_from_this());
        return true;
    }

    bool RemuxerBintr::SetBatchSize(uint batchSize)
    {
        LOG_FUNC();

        // This service will be called by the parent Branch just prior to LinkAll.
        // LinkAll(). We only update the Bintr's batch-size if not explicity set 
        // by the client, i.e. when supporting dynamic Source add/remove. 
        if (m_batchSizeSetByClient)
        {
            LOG_INFO("Batch-size set by client for RemuxerBintr '"
                << GetName() << "' - not updating" );
        }
        else
        {
            m_batchSize = batchSize;
        }
        return true;
    }
    
    uint RemuxerBintr::GetBatchSize()
    {
        LOG_FUNC();

        return m_batchSize;
    }

    bool RemuxerBintr::OverrideBatchSize(uint batchSize)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update batch size for RemuxerBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_batchSize = batchSize;

        if (!batchSize)
        {
            LOG_INFO("batch-size cleared by client for RemuxerBintr '" 
                << GetName() << "'");
            m_batchSizeSetByClient = false;
        }
        else
        {
            m_batchSizeSetByClient = true;
        }
        return true;
    }

    const char* RemuxerBintr::GetStreammuxConfigFile(
        DSL_BINTR_PTR pChildComponent)
    {
        if (!IsChild(pChildComponent))
        {
            LOG_ERROR("Can't get config-file for Branch '" 
                << pChildComponent->GetName() 
                << "' as it's not a child of Remuxer '"
                << GetName() << "'");
            return NULL;
        }
        return m_childBranches[pChildComponent->GetName()]->
            GetStreammuxConfigFile();
    }
    
    bool RemuxerBintr::SetStreammuxConfigFile(DSL_BINTR_PTR pChildComponent,
            const char* configFile)
    {
        LOG_FUNC();

        if (!IsChild(pChildComponent))
        {
            LOG_ERROR("Can't set config-file for Branch '" 
                << pChildComponent->GetName() 
                << "' as it's not a child of Remuxer '"
                << GetName() << "'");
            return false;
        }
        
        return m_childBranches[pChildComponent->GetName()]->
            SetStreammuxConfigFile(configFile);
    }

    void RemuxerBintr::GetBatchProperties(uint* batchSize, 
        int* batchTimeout)
    {
        LOG_FUNC();

        *batchSize = m_batchSize;
        *batchTimeout = m_batchTimeout;
    }

    bool RemuxerBintr::SetBatchProperties(uint batchSize, 
        int batchTimeout)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't update batch properties for RemuxerBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_batchTimeout = batchTimeout;
        m_batchSize = batchSize;

        if (!batchSize)
        {
            LOG_INFO("batch-size cleared by client for RemuxerBintr '" 
                << GetName() << "'");
            m_batchSizeSetByClient = false;
        }
        else
        {
            m_batchSizeSetByClient = true;
        }
        return true;
    }

    void RemuxerBintr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();

        *width = m_width;
        *height = m_height;
    }

    bool RemuxerBintr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("Can't set Streammuxer dimension for RemuxerBintr '" 
                << GetName() << "' as it is currently linked");
            return false;
        }
        m_width = width;
        m_height = height;
        return true;
    }

    bool RemuxerBintr::SetNvbufMemType(uint nvbufMemType)
    {
        LOG_FUNC();
        
        if (m_useNewStreammux)
        {
            LOG_ERROR("Can't update nvbuf-memory-type for RemuxerBintr '" 
                << GetName() << "' USE_NEW_NVSTREAMMUX must be set to 'yes'");
            return false;
        }
        if (m_isLinked)
        {
            LOG_ERROR("Unable to set NVIDIA buffer memory type for RemuxerBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_nvbufMemType = nvbufMemType;

        return true;
    }

    bool RemuxerBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();

        if (m_useNewStreammux)
        {
            LOG_ERROR("Can't update gpu-id for RemuxerBintr '" 
                << GetName() << "' USE_NEW_NVSTREAMMUX must be set to 'yes'");
            return false;
        }
        if (m_isLinked)
        {
            LOG_ERROR("Unable to set GPU ID for RemuxerBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }

        m_gpuId = gpuId;

        return true;
    }
}