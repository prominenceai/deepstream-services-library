/*
The MIT License

Copyright (c) 2019-2024, Prominence AI, Inc.

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
    {
        LOG_FUNC();

        if (!SetStreammuxEnabled(DSL_VIDEOMUX, true))
        {
            throw std::exception();
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

        // Ensure that the correct Streammux is enabled for this type of Source        
        if (!GetStreammuxEnabled(DSL_VIDEOMUX) and 
            !GetStreammuxEnabled(DSL_AUDIOMUX))
        {
            LOG_ERROR("Can't add Source '" << pChildSource->GetName() 
                << "'. The Pipeline's Audiomux and Videomux are currently disabled'");
            return false;
        }
        if ((pChildSource->GetMediaType() == DSL_MEDIA_TYPE_AUDIO_ONLY) and
            !GetStreammuxEnabled(DSL_AUDIOMUX))
        {
            LOG_ERROR("Can't add audio-only Source '" << pChildSource->GetName() 
                << "' The Pipeline's Audiomux is currently disabled'");
            return false;
        }
        if ((pChildSource->GetMediaType() == DSL_MEDIA_TYPE_VIDEO_ONLY) and
            !GetStreammuxEnabled(DSL_VIDEOMUX))
        {
            LOG_ERROR("Can't add video-only Source '" << pChildSource->GetName() 
                << "' The Pipeline's Videomux is currently disabled'");
            return false;
        }

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
        
        uint padId(0);
        
        // find the next available unused stream-id
        auto ivec = find(pVideomux->m_usedRequestPadIds.begin(), 
            pVideomux->m_usedRequestPadIds.end(), false);
        
        // If we're inserting into the location of a previously remved source
        if (ivec != pVideomux->m_usedRequestPadIds.end())
        {
            padId = ivec - pVideomux->m_usedRequestPadIds.begin();
            pVideomux->m_usedRequestPadIds[padId] = true;
        }
        // Else we're adding to the end of th indexed map
        else
        {
            padId = pVideomux->m_usedRequestPadIds.size();
            pVideomux->m_usedRequestPadIds.push_back(true);
        }            
        // Set the source's request sink pad-id
        pChildSource->SetVideoRequestPadId(padId);

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
                !pChildSource->LinkToSinkMuxer(pVideomux->Get(),
                sinkPadName.c_str()))
            {
                LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                    << "' failed to Link Child Source '" 
                    << pChildSource->GetName() << "'");
                return false;
            }
            if (!pVideomux->m_batchSizeSetByClient)
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
            LOG_INFO("Unlinking " << &pVideomux->GetName() << " from " 
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

            if (!pVideomux->m_batchSizeSetByClient)
            {
                // Decrement the current batch-size
                m_batchSize--;
            }
        }
        // Erase the Source the source from the name<->unique-id database.
        Services::GetServices()->_sourceNameErase(pChildSource->GetCStrName());
        
        // unreference and remove from the child source collections
        m_pChildSources.erase(pChildSource->GetName()); 
        m_pChildSourcesIndexed.erase(pChildSource->GetVideoRequestPadId());

        // set the used-stream id as available for reuse
        pVideomux->m_usedRequestPadIds[pChildSource->GetVideoRequestPadId()] = false;
        pChildSource->SetVideoRequestPadId(-1);
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
                "sink_" + std::to_string(imap.second->GetVideoRequestPadId());
            
            if (!imap.second->LinkAll() or 
                !imap.second->LinkToSinkMuxer(pVideomux->Get(),
                    sinkPadName.c_str()))
            {
                LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                    << "' failed to Link Child Source '" 
                    << imap.second->GetName() << "'");
                return false;
            }
            // If we're linking an RTSP Source, determine if EOS consumer 
            // should be added to Streammuxer
            if (imap.second->IsType(typeid(RtspSourceBintr)) 
                and !pVideomux->HasEosConsumer())
            {
                DSL_RTSP_SOURCE_PTR pRtspSource = 
                    std::dynamic_pointer_cast<RtspSourceBintr>(imap.second);
                
                // If stream management is enabled for at least one RTSP source, 
                // add the EOS Consumer
                if (pRtspSource->GetBufferTimeout())
                {
                    LOG_INFO("Adding EOS Consumer to Streammuxer 'src' pad on first RTSP Source");
                    
                    // Create the Pad Probe and EOS Consumer to drop the EOS event that 
                    // occurs on loss of RTSP stream, allowing the Pipeline to continue 
                    // to play. Each RTSP source will then manage their own restart 
                    // attempts and time management.

                    pVideomux->AddEosConsumer();
                }
            }

        }
        // Set the Batch size to the nuber of sources owned if not already set
        if (!pVideomux->m_batchSizeSetByClient)
        {
            m_batchSize = m_pChildSources.size();
            pVideomux->Get()->SetAttribute("batch-size", m_batchSize);
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
            LOG_INFO("Unlinking " << &pVideomux->GetName() 
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
        if (!pVideomux->m_batchSizeSetByClient)
        {
            m_batchSize = 0;
        }
        m_isLinked = false;
    }

    boolean PipelineSourcesBintr::GetStreammuxEnabled(streammux_type streammux)
    {
        LOG_FUNC();

        DSL_STREAMMUX_PTR pStreammux = (streammux == DSL_VIDEOMUX)
            ? pVideomux
            : pAudiomux;
    
        return (boolean)(pStreammux != nullptr);
    }

    bool PipelineSourcesBintr::SetStreammuxEnabled(streammux_type streammux, 
        boolean enabled)
    {
        LOG_FUNC();

        if (m_pChildSources.size())
        {
            LOG_ERROR("Can't update enabled property for StreammuxBintr '" 
                << GetName() << "' after Sources have been added");
            return false;
        }
        DSL_STREAMMUX_PTR pStreammux = (streammux == DSL_VIDEOMUX)
            ? pVideomux
            : pAudiomux;

        if (enabled == (boolean)(pStreammux != nullptr))
        {
            LOG_ERROR("Can't set enabled property for StreammuxBintr '" 
                << GetName() << "' to its current state of " << enabled);
            return false;
        }
        if (streammux == DSL_VIDEOMUX)
        {
            pVideomux = (enabled)
                ? DSL_STREAMMUX_NEW("video-streammux-", 
                    GetGstObject(), m_uniquePipelineId, "video_src")
                : nullptr;
        }
        else
        {
            pAudiomux = (enabled)
                ? DSL_STREAMMUX_NEW("audio-streammux-", 
                    GetGstObject(), m_uniquePipelineId, "audio_src")
                : nullptr;
        }
        return true;
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
            LOG_ERROR("Can't update live-source property for StreammuxBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_areSourcesLive = isLive;

        if (GetStreammuxEnabled(DSL_AUDIOMUX))
        {
            pAudiomux->PlayTypeIsLiveSet(isLive);    
        }
        if (GetStreammuxEnabled(DSL_VIDEOMUX))
        {
            pVideomux->PlayTypeIsLiveSet(isLive);
        }
        return true;
    }

    void PipelineSourcesBintr::EosAll()
    {
        LOG_FUNC();
        
        // Send EOS message to each source object.
        for (auto const& imap: m_pChildSources)
        {
            LOG_INFO("Sending EOS for Source "  << imap.second->GetName());
            imap.second->NullSrcEosSinkMuxer();
            // gst_element_send_event(imap.second->GetGstElement(), 
            //     gst_event_new_eos());
        }
    }

    void PipelineSourcesBintr::DisableEosConsumers()
    {
        // Call on all Sources to disable their EOS consumer if one
        // has been added.
        for (auto const& imap: m_pChildSources)
        {
            imap.second->DisableEosConsumer();
        }
        // Call on the Streammuxer to do the same.

        pVideomux->RemoveEosConsumer();
    }


}
