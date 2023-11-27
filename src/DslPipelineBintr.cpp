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
#include "DslSurfaceTransform.h"
#include "DslServices.h"
#include "DslPipelineBintr.h"

namespace DSL
{
    // Initialize the global/static vector of used pipeline-ids.
    std::vector<bool> PipelineBintr::m_usedPipelineIds;
    
    PipelineBintr::PipelineBintr(const char* name)
        : BranchBintr(name, true)      // Pipeline = true
        , PipelineStateMgr(m_pGstObj)
        , PipelineBusSyncMgr(m_pGstObj)
    {
        LOG_FUNC();

        // find the next available unused pipeline-id
        auto ivec = find(m_usedPipelineIds.begin(), m_usedPipelineIds.end(), false);
        
        // If we're inserting into the location of a previously remved source
        if (ivec != m_usedPipelineIds.end())
        {
            m_pipelineId = ivec - m_usedPipelineIds.begin();
            m_usedPipelineIds[m_pipelineId] = true;
        }
        // Else we're adding to the end of the vector
        else
        {
            m_pipelineId = m_usedPipelineIds.size(); // 0 based
            m_usedPipelineIds.push_back(true);
        }            

        // Instantiate the PipelineSourcesBintr for the Pipeline Bintr, 
        std::string sourcesBinName = GetName() + "-sources-bin";
        m_pPipelineSourcesBintr = 
            DSL_PIPELINE_SOURCES_NEW(sourcesBinName.c_str(), m_pipelineId);

        // Add PipelineSourcesBintr as chid of this PipelineBintr.
        AddChild(m_pPipelineSourcesBintr);
    }

    PipelineBintr::~PipelineBintr()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            Stop();
        }
        // clear the pipeline-id for reuse.
        m_usedPipelineIds[m_pipelineId] = false;
    }

    bool PipelineBintr::AddSourceBintr(DSL_BASE_PTR pSourceBintr)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr->
            AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr)))
        {
            return false;
        }
        return true;
    }

    bool PipelineBintr::IsSourceBintrChild(DSL_BASE_PTR pSourceBintr)
    {
        LOG_FUNC();

        return (m_pPipelineSourcesBintr->
            IsChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr)));
    }

    bool PipelineBintr::RemoveSourceBintr(DSL_BASE_PTR pSourceBintr)
    {
        LOG_FUNC();

        // Must cast to SourceBintr first so that correct Instance of 
        // RemoveChild is called
        return m_pPipelineSourcesBintr->
            RemoveChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr));
    }

    void PipelineBintr::GetStreammuxBatchProperties(uint* batchSize, 
        int* batchTimeout)
    {
        LOG_FUNC();

        m_pPipelineSourcesBintr->
            GetStreammuxBatchProperties(batchSize, batchTimeout);
    }

    bool PipelineBintr::SetStreammuxBatchProperties(uint batchSize, 
        int batchTimeout)
    {
        LOG_FUNC();

        return m_pPipelineSourcesBintr->
            SetStreammuxBatchProperties(batchSize, batchTimeout);
    }

    uint PipelineBintr::GetStreammuxNumSurfacesPerFrame()
    {
        LOG_FUNC();

        return m_pPipelineSourcesBintr->GetStreammuxNumSurfacesPerFrame();
    }
    
    bool PipelineBintr::SetStreammuxNumSurfacesPerFrame(uint num)
    {
        LOG_FUNC();

        return m_pPipelineSourcesBintr->SetStreammuxNumSurfacesPerFrame(num);
    }
    
    bool PipelineBintr::GetStreammuxSyncInputsEnabled()
    {
        LOG_FUNC();

        return m_pPipelineSourcesBintr->GetStreammuxSyncInputsEnabled();
    }
    
    bool PipelineBintr::SetStreammuxSyncInputsEnabled(boolean enabled)
    {
        LOG_FUNC();

        return m_pPipelineSourcesBintr->SetStreammuxSyncInputsEnabled(enabled);
    }
    
    bool PipelineBintr::AddStreammuxTiler(DSL_BASE_PTR pTilerBintr)
    {
        if (m_pStreammuxTilerBintr)
        {
            LOG_INFO("Pipeline '" << GetName() 
                << "' already has a Tiler attached to the Stream-Muxer's ouput");
            return false;
        }
        if (m_isLinked)
        {
            LOG_INFO("Can't add a Tiler to the Stream-Muxer output for Pipeline '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_pStreammuxTilerBintr = std::dynamic_pointer_cast<TilerBintr>(pTilerBintr);
        return AddChild(m_pStreammuxTilerBintr);
    }
    
    bool PipelineBintr::RemoveStreammuxTiler()
    {
        if (!m_pStreammuxTilerBintr)
        {
            LOG_INFO("Pipeline '" << GetName() 
                << "' does not have a Tiler attached to the Stream-Muxer's ouput");
            return false;
        }
        if (m_isLinked)
        {
            LOG_INFO("Can't remove a Tiler from the Stream-Muxer output for Pipeline '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        RemoveChild(m_pStreammuxTilerBintr);
        m_pStreammuxTilerBintr = nullptr;
        return true;
    }
    
    bool PipelineBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_INFO("Components for Pipeline '" << GetName() << "' are already assembled");
            return false;
        }
        if (!m_pPipelineSourcesBintr->GetNumChildren())
        {
            LOG_ERROR("Pipline '" << GetName() << "' has no required Source component - and is unable to link");
            return false;
        }

        
        // Start with an empty list of linked components
        m_linkedComponents.clear();

        // Link all Source Elementrs (required component), and all Sources to the Streammuxer
        // then add the PipelineSourcesBintr as the Source (head) component for this Pipeline
        if (!m_pPipelineSourcesBintr->LinkAll())
        {
            return false;
        }
        m_linkedComponents.push_back(m_pPipelineSourcesBintr);
        
        LOG_INFO("Pipeline '" << GetName() << "' Linked up all Source '" << 
            m_pPipelineSourcesBintr->GetName() << "' successfully");

        int batchTimeout(0); // we don't care about batch-timeout
        GetStreammuxBatchProperties(&m_batchSize, &batchTimeout);

        if (m_pStreammuxTilerBintr)
        {
            // Link All Tiler Elementrs and add as the next component in the Branch
            m_pStreammuxTilerBintr->SetBatchSize(m_batchSize);
            if (!m_pStreammuxTilerBintr->LinkAll() or
                !m_linkedComponents.back()->LinkToSink(m_pStreammuxTilerBintr))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pStreammuxTilerBintr);
            LOG_INFO("Pipeline '" << GetName() << "' Linked up Tiler '" << 
                m_pStreammuxTilerBintr->GetName() 
                << "' to the Streammuxer output successfully");
        }

        // call the base class to Link all remaining components.
        return BranchBintr::LinkAll();
    }

    bool PipelineBintr::Play()
    {
        LOG_FUNC();
        
        GstState currentState;
        GetState(currentState, 0);
        if (currentState == GST_STATE_NULL or currentState == GST_STATE_READY)
        {
            if (!LinkAll())
            {
                LOG_ERROR("Unable to prepare Pipeline '" << GetName() << "' for Play");
                return false;
            }
            // For non-live sources we Pause to preroll before we play
            if (!m_pPipelineSourcesBintr->StreammuxPlayTypeIsLiveGet())
            {
                if (!SetState(GST_STATE_PAUSED, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
                {
                    LOG_ERROR("Failed to Pause non-live soures before playing Pipeline '" << GetName() << "'");
                    return false;
                }
            }
        }
                
        // Call the base class to complete the Play process
        return SetState(GST_STATE_PLAYING, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND);
    }

    bool PipelineBintr::Pause()
    {
        LOG_FUNC();
        
        GstState state;
        GetState(state, 0);
        if (state != GST_STATE_PLAYING)
        {
            LOG_WARN("Pipeline '" << GetName() << "' is not in a state of Playing");
            return false;
        }
        
        // Call the base class to Pause the Pipeline - can be called from any context.
        if (!SetState(GST_STATE_PAUSED, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
        {
            LOG_ERROR("Failed to Pause Pipeline '" << GetName() << "'");
            return false;
        }
        return true;
    }

    bool PipelineBintr::Stop()
    {
        LOG_FUNC();
        
        if (!IsLinked())
        {
            LOG_WARN("Pipeline '" << GetName() << "' is not linked");
            return true;
            // return false;
        }
        GstState state;
        GetState(state, 0);
        if (state == GST_STATE_PAUSED)
        {
            LOG_INFO("Setting Pipeline '" << GetName() 
                << "' to PLAYING before setting to NULL");
            // Call the base class to Pause the Pipeline - can be called from any context.
            if (!SetState(GST_STATE_PLAYING, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
            {
                LOG_ERROR("Failed to set Pipeline '" << GetName() 
                    << "' to PLAYING before setting to NULL");
                return false;
            }
        }
        
        // Need to check the context to see if we're running from either
        // the XDisplay thread or the bus-watch fucntion
        
        // Try and lock the shared-client-mutex first
        if (!g_mutex_trylock(&*m_pSharedClientCbMutex))
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_busWatchMutex);
            // lock-failed which means we are already in the XWindow thread context
            // calling on a client handler function for Key release or xWindow delete. 
            // Safe to stop the Pipeline in this context.
            LOG_INFO("dsl_pipeline_stop called from client-callback context");
            HandleStop();
            return true;
        }
        // Try the bus-watch mutex next
        if (!g_mutex_trylock(&m_busWatchMutex))
        {
            // lock-failed which means we're in the bus-watch function context
            // calling on a client listener or handler function. Safe to stop 
            // the Pipeline in this context. 
            LOG_INFO("dsl_pipeline_stop called from bus-watch-function thread context");
            HandleStop();
            g_mutex_unlock(&*m_pSharedClientCbMutex);
            return true;
        }
        // If the main loop is running -- normal case -- then we can't change the 
        // state of the Pipeline in the Application's context. 
        if ((m_pMainLoop and g_main_loop_is_running(m_pMainLoop)) or
            (!m_pMainLoop and g_main_loop_is_running(
                DSL::Services::GetServices()->GetMainLoopHandle())))
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommsMutex);
            LOG_INFO("Sending application message to stop the pipeline");
            
            gst_element_post_message(GetGstElement(),
                gst_message_new_application(GetGstObject(),
                    gst_structure_new_empty("stop-pipline")));

            g_mutex_unlock(&*m_pSharedClientCbMutex);
            g_mutex_unlock(&m_busWatchMutex);
                    
            // We need a timeout in case the condition is never met/cleared
            gint64 endtime = g_get_monotonic_time () + 
                (DSL_DEFAULT_WAIT_FOR_EOS_TIMEOUT_IN_SEC *2 * G_TIME_SPAN_SECOND);
            if (!g_cond_wait_until(&m_asyncCommsCond, &m_asyncCommsMutex, endtime))
            {
                LOG_WARN("Pipeline '" << GetName() 
                    << "' failed to complete async-stop");
                return false;
            }
            else
            {
                return true;
            }
        }
        // Else, client has stopped the main-loop or we are running under test 
        // without the mainloop running - can't send a message so handle stop now.
        else
        {
            HandleStop();
        }
        g_mutex_unlock(&*m_pSharedClientCbMutex);
        g_mutex_unlock(&m_busWatchMutex);
        return true;
    }

    void PipelineBintr::HandleStop()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommsMutex);
        
        // Call on all sources to disable their EOS consumers, before sending EOS
        m_pPipelineSourcesBintr->DisableEosConsumers();
        
        // If the client is not stoping due to EOS, we must EOS the Pipeline 
        // to gracefully stop any recording in progress before changing the 
        // Pipeline's state to NULL, 
        if (!m_eosFlag)
        {
            m_eosFlag = true;
            
            // Send an EOS event to the Pipline bin. 
            SendEos();
            
            // once the EOS event has been received on all sink pads of all
            // elements, an EOS message will be posted on the bus. We need to
            // discard all bus messages while waiting for the EOS message.
            GstMessage* msg = gst_bus_timed_pop_filtered(m_pGstBus, 
                DSL_DEFAULT_WAIT_FOR_EOS_TIMEOUT_IN_SEC * GST_SECOND,
                    (GstMessageType)(GST_MESSAGE_CLOCK_LOST | GST_MESSAGE_ERROR | 
                        GST_MESSAGE_EOS));

            if (!msg or GST_MESSAGE_TYPE(msg) != GST_MESSAGE_EOS)
            {
                LOG_WARN("Pipeline '" << GetName() 
                    << "' failed to receive final EOS message on dsl_pipeline_stop");
            }
            else
            {
                LOG_INFO("Pipeline '" << GetName() 
                    << "' completed async-stop successfully");
            }
        }

        if (!SetState(GST_STATE_NULL, 
            DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
        {
            LOG_ERROR("Failed to Stop Pipeline '" << GetName() << "'");
        }
        
        m_eosFlag = false;
        UnlinkAll();
        
        g_cond_signal(&m_asyncCommsCond);
    }

    bool PipelineBintr::IsLive()
    {
        LOG_FUNC();
        
        if (!m_pPipelineSourcesBintr)
        {
            LOG_INFO("Pipeline '" << GetName() 
                << "' has no sources, therefore is-live = false");
            return false;
        }
        return m_pPipelineSourcesBintr->StreammuxPlayTypeIsLiveGet();
    }
    
    void PipelineBintr::DumpToDot(char* filename)
    {
        LOG_FUNC();
        
        GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(m_pGstObj), 
            GST_DEBUG_GRAPH_SHOW_ALL, filename);
    }
    
    void PipelineBintr::DumpToDotWithTs(char* filename)
    {
        LOG_FUNC();
        
        GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN(m_pGstObj), 
            GST_DEBUG_GRAPH_SHOW_ALL, filename);
    }

    static int PipelineStop(gpointer pPipeline)
    {
        static_cast<PipelineBintr*>(pPipeline)->HandleStop();
        
        // Return false to self destroy timer - one shot.
        return false;
    }

} // DSL
