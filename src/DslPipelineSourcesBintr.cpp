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
#include "DslPipelineSourcesBintr.h"
#include "DslServices.h"

namespace DSL
{
    PipelineSourcesBintr::PipelineSourcesBintr(const char* name,
        uint uniquePipelineId)
        : Bintr(name)
        , m_uniquePipelineId(uniquePipelineId)
        , m_numAudioSources(0)
        , m_numVideoSources(0)
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
            !GetStreammuxEnabled(DSL_AUDIOMUX) and
            !GetAudiomixEnabled())
        {
            LOG_ERROR("Can't add Source '" << pChildSource->GetName() 
                << "'. The Pipeline's Audiomuxer, Videomuxer, and Audiomixer are all disabled'");
            return false;
        }
        if ((pChildSource->GetMediaType() == DSL_MEDIA_TYPE_AUDIO_ONLY) and
            (!GetStreammuxEnabled(DSL_AUDIOMUX) and !GetAudiomixEnabled()))
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

        // Update the number of child Sources based on media type.
        if (pChildSource->GetMediaType() & DSL_MEDIA_TYPE_AUDIO_ONLY)
        {
            m_numAudioSources++;
        }
        if (pChildSource->GetMediaType() & DSL_MEDIA_TYPE_VIDEO_ONLY)
        {
            m_numVideoSources++;
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
        
        // If the Pipeline is currently in a linked state, linkAll Elementrs now and
        // link to the Video and/or Audio Streammuxers
        if (IsLinked())
        {
            if (!pChildSource->LinkAll() or
                !LinkChildToSinkMuxers(pChildSource))
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
            LOG_INFO("Unlinking " << &m_pVideomux->GetName() << " from " 
                << pChildSource->GetName());
                
            // unlink the source from the Streammuxers
            if (!UnlinkChildFromSinkMuxers(pChildSource))
            {
                return false;
            }
            // unlink all of the ChildSource's Elementrs
            pChildSource->UnlinkAll();
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

        // Update the number of child Sources based on media type.
        if (pChildSource->GetMediaType() & DSL_MEDIA_TYPE_AUDIO_ONLY)
        {
            m_numAudioSources--;
        }
        if (pChildSource->GetMediaType() & DSL_MEDIA_TYPE_VIDEO_ONLY)
        {
            m_numVideoSources--;
        }

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
            if (!imap.second->LinkAll() or
                !LinkChildToSinkMuxers(imap.second))
            {
                return false;
            }
            // If we're linking an RTSP Source, determine if EOS consumer 
            // should be added to Streammuxer
            if (imap.second->IsType(typeid(RtspSourceBintr)) 
                and !m_pVideomux->HasEosConsumer())
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

                    if (GetStreammuxEnabled(DSL_AUDIOMUX))
                    {
                        m_pAudiomux->AddEosConsumer();    
                    }
                    if (GetStreammuxEnabled(DSL_VIDEOMUX))
                    {
                        m_pVideomux->AddEosConsumer();
                    }
                }
            }
        }
        SetBatchSizes();
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
            if (!UnlinkChildFromSinkMuxers(imap.second))
            {
                return;
            }
            
            // unlink all of the ChildSource's Elementrs
            imap.second->UnlinkAll();
        }
        ClearBatchSizes();
        m_isLinked = false;
    }

    void PipelineSourcesBintr::SetBatchSizes()
    {
        LOG_FUNC();

            // Set the Batch size to the nuber of sources owned if not already set
        if (m_pAudiomux)
        {
            if (!m_pAudiomux->m_batchSizeSetByClient)
            {
                m_audioBatchSize = m_numAudioSources;
                m_pAudiomux->Get()->SetAttribute("batch-size", m_audioBatchSize);
            }
            else
            {
                m_audioBatchSize = m_pAudiomux->GetBatchSize();
            }
        }
        if (m_pVideomux)
        {
            if (!m_pVideomux->m_batchSizeSetByClient)
            {
                m_videoBatchSize = m_numVideoSources;
                m_pVideomux->Get()->SetAttribute("batch-size", m_videoBatchSize);
            }
            else
            {
                m_videoBatchSize = m_pVideomux->GetBatchSize();
            }
        }
    }

    void PipelineSourcesBintr::ClearBatchSizes()
    {
        LOG_FUNC();

        m_audioBatchSize = 0;
        m_videoBatchSize = 0;
    }

    bool PipelineSourcesBintr::LinkChildToSinkMuxers(DSL_SOURCE_PTR pChildSource)
    {
        LOG_FUNC();
        
        // If the Source supports Audio and the Audio Streammux is enabled 
        if (pChildSource->GetMediaType() | DSL_MEDIA_TYPE_AUDIO_ONLY)
        {
            if (GetStreammuxEnabled(DSL_AUDIOMUX))
            {
                std::string sinkPadName = 
                    "sink_" + std::to_string(pChildSource->GetRequestPadId());

                if (!pChildSource->LinkToSinkMuxer(m_pAudiomux->Get(),
                    "audio_src", sinkPadName.c_str()))
                {
                    LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                        << "' failed to link audio for Child Source '" 
                        << pChildSource->GetName() << "'");
                    return false;
                }
            }
            if (GetAudiomixEnabled())
            {
                if (!pChildSource->LinkToSinkMuxer(m_pAudiomix->Get(),
                    "audio_src", "sink_%u"))
                {
                    LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                        << "' failed to link audio for Child Source '" 
                        << pChildSource->GetName() << "'");
                    return false;
                }
                DSL_AUDIO_SOURCE_PTR pAudioSource = 
                    std::dynamic_pointer_cast<AudioSourceBintr>(pChildSource);

                // Set the mute and volume pad-properties to the current values
                // as they may have been updated by the client before linking. 
                if (!SetAudiomixMuteEnabled(pAudioSource, 
                    pAudioSource->GetAudiomixMuteEnabled())
                    or !SetAudiomixVolume(pAudioSource, 
                    pAudioSource->GetAudiomixVolume()))
                {
                    return false;
                }
            }            
        }
        // If the Source supports Video and the Vidio Streammux is enabled 
        if ((pChildSource->GetMediaType() | DSL_MEDIA_TYPE_VIDEO_ONLY) and 
            GetStreammuxEnabled(DSL_VIDEOMUX))
        {
            std::string sinkPadName = 
                "sink_" + std::to_string(pChildSource->GetRequestPadId());

            if (!pChildSource->LinkToSinkMuxer(m_pVideomux->Get(),
                "video_src", sinkPadName.c_str()))
            {
                LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                    << "' failed to link video for Child Source '" 
                    << pChildSource->GetName() << "'");
                return false;
            }
        }
        return true;
    }

    bool PipelineSourcesBintr::UnlinkChildFromSinkMuxers(DSL_SOURCE_PTR pChildSource)
    {
        LOG_FUNC();
        
        // If the Source supports Audio and the Audio Streammux is enabled 
        if (pChildSource->GetMediaType() | DSL_MEDIA_TYPE_AUDIO_ONLY) 
        {
            if (GetStreammuxEnabled(DSL_AUDIOMUX))
            {
                LOG_INFO("Unlinking child Source '" << pChildSource->GetName() 
                    << "' from AudioMuxer");

                if (!pChildSource->UnlinkFromSinkMuxer(m_pAudiomux->Get(),
                    "audio_src"))
                {
                    LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                        << "' failed to unlink audio for Child Source '" 
                        << pChildSource->GetName() << "'");
                    return false;
                }
            }
            if (GetAudiomixEnabled())
            {
                if (!pChildSource->UnlinkFromSinkMuxer(m_pAudiomix->Get(),
                    "audio_src"))
                {
                    LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                        << "' failed to link audio for Child Source '" 
                        << pChildSource->GetName() << "'");
                    return false;
                }
            }            
        }
        // If the Source supports Video and the Vidio Streammux is enabled 
        if ((pChildSource->GetMediaType() | DSL_MEDIA_TYPE_VIDEO_ONLY) and 
            GetStreammuxEnabled(DSL_VIDEOMUX))
        {
            LOG_INFO("Unlinking child Source '" << pChildSource->GetName() 
                << "' from VideoMuxer");
                
            if (!pChildSource->UnlinkFromSinkMuxer(m_pVideomux->Get(),
                "video_src"))
            {
                LOG_ERROR("PipelineSourcesBintr '" << GetName() 
                    << "' failed to unlink video for Child Source '" 
                    << pChildSource->GetName() << "'");
                return false;
            }
        }
        return true;
    }

    bool PipelineSourcesBintr::GetAudiomixMuteEnabled(
        DSL_AUDIO_SOURCE_PTR pChildSource, boolean* enabled)
    {
        LOG_FUNC();
        
        // Check for the relationship first
        if (!IsChild(pChildSource))
        {
            LOG_ERROR("Source '" << pChildSource->GetName() 
                << "' is not a child of '" << GetName() << "'");
            return false;
        }
        if (!GetAudiomixEnabled())
        {
            LOG_ERROR("The Audiomuxer is not enabled for '" << GetName() << "'");
            return false;
        }

        *enabled = pChildSource->GetAudiomixMuteEnabled();

        // call the parent class to complete the link-to-sink
        return true;
    }

    bool PipelineSourcesBintr::SetAudiomixMuteEnabled(
        DSL_AUDIO_SOURCE_PTR pChildSource, boolean enabled)
    {
        LOG_FUNC();
        
        // Check for the relationship first
        if (!IsChild(pChildSource))
        {
            LOG_ERROR("Source '" << pChildSource->GetName() 
                << "' is not a child of '" << GetName() << "'");
            return false;
        }
        if (!GetAudiomixEnabled())
        {
            LOG_ERROR("The Audiomuxer is not enabled for '" << GetName() << "'");
            return false;
        }
        
        // If we're currently linked, updated the pad property
        if (m_isLinked)
        {
            // Get a reference to the Source's audio-src pad
            GstPad* pStaticSrcPad = gst_element_get_static_pad(
                pChildSource->GetGstElement(), "audio_src");
            if (!pStaticSrcPad)
            {
                LOG_ERROR("Failed to get static source pad for GstNodetr '" 
                    << pChildSource->GetName() << "'");
                return false;
            }
            
            // Get a reference to the Mixer's sink pad that is connected
            // to the Source's pad
            GstPad* pRequestedSinkPad = gst_pad_get_peer(pStaticSrcPad);
            if (!pRequestedSinkPad)
            {
                LOG_ERROR("Failed to get requested sink pad peer for GstNodetr '" 
                    << pChildSource->GetName() << "'");
                return false;
            }
            g_object_set(pRequestedSinkPad, "mute", enabled, NULL);

            // unreference both the static source pad and requested sink
            gst_object_unref(pStaticSrcPad);
            gst_object_unref(pRequestedSinkPad);
        }
        // Call on the Source to save the current enabled setting
        pChildSource->SetAudiomixMuteEnabled(enabled);

        // call the parent class to complete the link-to-sink
        return true;
    }

    bool PipelineSourcesBintr::GetAudiomixVolume(
        DSL_AUDIO_SOURCE_PTR pChildSource, double* volume)
    {
        LOG_FUNC();
        
        // Check for the relationship first
        if (!IsChild(pChildSource))
        {
            LOG_ERROR("Source '" << pChildSource->GetName() 
                << "' is not a child of '" << GetName() << "'");
            return false;
        }
        if (!GetAudiomixEnabled())
        {
            LOG_ERROR("The Audiomuxer is not enabled for '" << GetName() << "'");
            return false;
        }

        // Get the current value saved by Audio Source
        *volume = pChildSource->GetAudiomixVolume();

        // call the parent class to complete the link-to-sink
        return true;
    }

    bool PipelineSourcesBintr::SetAudiomixVolume(
        DSL_AUDIO_SOURCE_PTR pChildSource, double volume)
    {
        LOG_FUNC();
        
        // Check for the relationship first
        if (!IsChild(pChildSource))
        {
            LOG_ERROR("Source '" << pChildSource->GetName() 
                << "' is not a child of '" << GetName() << "'");
            return false;
        }
        if (!GetAudiomixEnabled())
        {
            LOG_ERROR("The Audiomuxer is not enabled for '" << GetName() << "'");
            return false;
        }

        // If we're currently linked, updated the pad property
        if (m_isLinked)
        {
            // Get a reference to the Source's audio-src pad
            GstPad* pStaticSrcPad = gst_element_get_static_pad(
                pChildSource->GetGstElement(), "audio_src");
            if (!pStaticSrcPad)
            {
                LOG_ERROR("Failed to get static source pad for GstNodetr '" 
                    << pChildSource->GetName() << "'");
                return false;
            }
            
            // Get a reference to the Mixer's sink pad that is connected
            // to the Source's pad
            GstPad* pRequestedSinkPad = gst_pad_get_peer(pStaticSrcPad);
            if (!pRequestedSinkPad)
            {
                LOG_ERROR("Failed to get requested sink pad peer for GstNodetr '" 
                    << pChildSource->GetName() << "'");
                return false;
            }
            g_object_set(pRequestedSinkPad, "volume", volume, NULL);

            // unreference both the static source pad and requested sink pad
            gst_object_unref(pStaticSrcPad);
            gst_object_unref(pRequestedSinkPad);
        }

        // Call on the Source to save the current volume setting
        pChildSource->SetAudiomixVolume(volume);

        // call the parent class to complete the link-to-sink
        return true;
    }

    boolean PipelineSourcesBintr::GetStreammuxEnabled(streammux_type streammux)
    {
        LOG_FUNC();

        DSL_STREAMMUX_PTR pStreammux = (streammux == DSL_VIDEOMUX)
            ? m_pVideomux
            : m_pAudiomux;
    
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
            ? m_pVideomux
            : m_pAudiomux;

        if (enabled == (boolean)(pStreammux != nullptr))
        {
            LOG_ERROR("Can't set enabled property for StreammuxBintr '" 
                << GetName() << "' to its current state of " << enabled);
            return false;
        }
        if (streammux == DSL_VIDEOMUX)
        {
            std::string videoStreammuxName = GetName() + "-video";
            m_pVideomux = (enabled)
                ? DSL_STREAMMUX_NEW(videoStreammuxName.c_str(), 
                    GetGstObject(), m_uniquePipelineId, "video_src")
                : nullptr;
        }
        else
        {
            std::string audioStreammuxName = GetName() + "-audio";
            m_pAudiomux = (enabled)
                ? DSL_STREAMMUX_NEW(audioStreammuxName.c_str(), 
                    GetGstObject(), m_uniquePipelineId, "audio_src")
                : nullptr;
        }

        UpdateMediaType();

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
            m_pAudiomux->PlayTypeIsLiveSet(isLive);    
        }
        if (GetStreammuxEnabled(DSL_VIDEOMUX))
        {
            m_pVideomux->PlayTypeIsLiveSet(isLive);
        }
        return true;
    }

    boolean PipelineSourcesBintr::GetAudiomixEnabled()
    {
        LOG_FUNC();

        return (boolean)(m_pAudiomix != nullptr);
    }

    bool PipelineSourcesBintr::SetAudiomixEnabled(boolean enabled)
    {
        LOG_FUNC();

        if (m_pChildSources.size())
        {
            LOG_ERROR("Can't update enabled property for Audiomixer '" 
                << GetName() << "' after Sources have been added");
            return false;
        }
        if (enabled == (boolean)(m_pAudiomix != nullptr))
        {
            LOG_ERROR("Can't set enabled property for Audiomixer '" 
                << GetName() << "' to its current state of " << enabled);
            return false;
        }
        if (enabled and GetStreammuxEnabled(DSL_AUDIOMUX))
        {
            LOG_ERROR("Can't set enabled property for Audiomixer '" 
                << GetName() << "' Audiomuxer is currently enabled");
            return false;
        }
        m_pAudiomix = (enabled)
            ? DSL_AUDIOMIX_NEW(GetCStrName(), 
                GetGstObject(), "audio_src")
            : nullptr;

        // Update the media-type based on the enabled state of both muxers.
        UpdateMediaType();

        return true;
    }

    void PipelineSourcesBintr::EosAll()
    {
        LOG_FUNC();
        
        // Send EOS message to each source object.
        for (auto const& imap: m_pChildSources)
        {
            LOG_INFO("Sending EOS for Source "  << imap.second->GetName());
            
            // gst_element_send_event(imap.second->GetGstElement(), 
            //     gst_event_new_eos());
            // If the Source supports Video and the Vidio Streammux is enabled 
            if ((imap.second->GetMediaType() | DSL_MEDIA_TYPE_VIDEO_ONLY) and 
                GetStreammuxEnabled(DSL_VIDEOMUX))
            {
                imap.second->NullSrcEosSinkMuxer("video_src");
            }
            // If the Source supports Audio and the Audio Streammux is enabled 
            if ((imap.second->GetMediaType() | DSL_MEDIA_TYPE_AUDIO_ONLY) and 
                GetStreammuxEnabled(DSL_AUDIOMUX))
            {
                imap.second->NullSrcEosSinkMuxer("audio_src");
            }
        }
    }

    void PipelineSourcesBintr::DisableEosConsumers()
    {
        LOG_FUNC();
        
        // Call on all Sources to disable their EOS consumer if one
        // has been added.
        for (auto const& imap: m_pChildSources)
        {
            imap.second->DisableEosConsumer();
        }
        // Call on the Streammuxer to do the same.

        if (GetStreammuxEnabled(DSL_AUDIOMUX))
        {
            m_pAudiomux->RemoveEosConsumer();    
        }
        if (GetStreammuxEnabled(DSL_VIDEOMUX))
        {
            m_pVideomux->RemoveEosConsumer();
        }
    }

    void PipelineSourcesBintr::UpdateMediaType()
    {
        LOG_FUNC();
        
        // Update the media-type based on the enabled state of both muxers 
        // and audio-mixer.
        uint mediaType = 0;
        if (m_pVideomux)
        {
            mediaType = mediaType | DSL_MEDIA_TYPE_VIDEO_ONLY;
        }
        if (m_pAudiomux)
        {
            mediaType = mediaType | DSL_MEDIA_TYPE_AUDIO_ONLY;
        }
        if (m_pAudiomix)
        {
            mediaType = mediaType | DSL_MEDIA_TYPE_AUDIO_ONLY;
        }
        m_mediaType = mediaType;
        LOG_INFO("Media type for PipelineSourcesBintr set to " << m_mediaType);
    }

}
