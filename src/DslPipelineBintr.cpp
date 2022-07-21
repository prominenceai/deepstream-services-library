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
    PipelineBintr::PipelineBintr(const char* name)
        : BranchBintr(name, true) // Pipeline = true
        , PipelineStateMgr(m_pGstObj)
        , PipelineXWinMgr(m_pGstObj)
    {
        LOG_FUNC();

        m_pPipelineSourcesBintr = DSL_PIPELINE_SOURCES_NEW("sources-bin");
        AddChild(m_pPipelineSourcesBintr);

        g_mutex_init(&m_asyncCommMutex);
    }

    PipelineBintr::~PipelineBintr()
    {
        LOG_FUNC();
        
        GstState state;
        GetState(state, 0);
        if (m_isLinked)
        {
            Stop();
        }
        g_mutex_clear(&m_asyncCommMutex);
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

        if (!m_pPipelineSourcesBintr)
        {
            LOG_INFO("Pipeline '" << GetName() << "' has no Sources");
            return false;
        }
        return (m_pPipelineSourcesBintr->
            IsChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr)));
    }

    bool PipelineBintr::RemoveSourceBintr(DSL_BASE_PTR pSourceBintr)
    {
        LOG_FUNC();

        // Must cast to SourceBintr first so that correct Instance of RemoveChild is called
        return m_pPipelineSourcesBintr->
            RemoveChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr));
    }

    uint PipelineBintr::GetStreamMuxNvbufMemType()
    {
        LOG_FUNC();

        return m_pPipelineSourcesBintr->GetStreamMuxNvbufMemType();
    }

    bool PipelineBintr::SetStreamMuxNvbufMemType(uint type)
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("Pipeline '" << GetName() 
                << "' is currently Linked - cudadec memory type can not be updated");
            return false;
            
        }
        m_pPipelineSourcesBintr->SetStreamMuxNvbufMemType(type);
        
        return true;
    }

    void PipelineBintr::GetStreamMuxBatchProperties(guint* batchSize, 
        uint* batchTimeout)
    {
        LOG_FUNC();

        m_pPipelineSourcesBintr->
            GetStreamMuxBatchProperties(batchSize, batchTimeout);
    }

    bool PipelineBintr::SetStreamMuxBatchProperties(uint batchSize, uint batchTimeout)
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("Pipeline '" << GetName() 
                << "' is currently Linked - batch properties can not be updated");
            return false;
            
        }
        m_pPipelineSourcesBintr->
            SetStreamMuxBatchProperties(batchSize, batchTimeout);
        
        return true;
    }

    bool PipelineBintr::GetStreamMuxDimensions(uint* width, uint* height)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has no Sources or Stream Muxer");
            return false;
        }
        m_pPipelineSourcesBintr->GetStreamMuxDimensions(width, height);
        return true;
    }

    bool PipelineBintr::SetStreamMuxDimensions(uint width, uint height)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has no Sources or Stream Muxer");
            return false;
        }
        m_pPipelineSourcesBintr->SetStreamMuxDimensions(width, height);
        return true;
    }
    
    bool PipelineBintr::GetStreamMuxPadding(bool* enabled)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has no Sources or Stream Muxer");
            return false;
        }
        m_pPipelineSourcesBintr->GetStreamMuxPadding(enabled);
        return true;
    }
    
    bool PipelineBintr::SetStreamMuxPadding(bool enabled)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has no Sources or Stream Muxer");
            return false;
        }
        m_pPipelineSourcesBintr->SetStreamMuxPadding(enabled);
        return true;
    }
    
    bool PipelineBintr::GetStreamMuxNumSurfacesPerFrame(uint* num)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has no Sources or Stream Muxer");
            return false;
        }
        m_pPipelineSourcesBintr->GetStreamMuxNumSurfacesPerFrame(num);
        return true;
    }
    
    bool PipelineBintr::SetStreamMuxNumSurfacesPerFrame(uint num)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr)
        {
            LOG_ERROR("Pipeline '" << GetName() << "' has no Sources or Stream Muxer");
            return false;
        }
        m_pPipelineSourcesBintr->SetStreamMuxNumSurfacesPerFrame(num);
        return true;
    }
    
    bool PipelineBintr::AddStreamMuxTiler(DSL_BASE_PTR pTilerBintr)
    {
        if (m_pStreamMuxTilerBintr)
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
        m_pStreamMuxTilerBintr = std::dynamic_pointer_cast<TilerBintr>(pTilerBintr);
        return AddChild(m_pStreamMuxTilerBintr);
    }
    
    bool PipelineBintr::RemoveStreamMuxTiler()
    {
        if (!m_pStreamMuxTilerBintr)
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
        RemoveChild(m_pStreamMuxTilerBintr);
        m_pStreamMuxTilerBintr = nullptr;
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

        // Link all Source Elementrs (required component), and all Sources to the StreamMuxer
        // then add the PipelineSourcesBintr as the Source (head) component for this Pipeline
        if (!m_pPipelineSourcesBintr->LinkAll())
        {
            return false;
        }
        m_linkedComponents.push_back(m_pPipelineSourcesBintr);
        
        LOG_INFO("Pipeline '" << GetName() << "' Linked up all Source '" << 
            m_pPipelineSourcesBintr->GetName() << "' successfully");

        uint batchTimeout(0);
        GetStreamMuxBatchProperties(&m_batchSize, &batchTimeout);

        if (m_pStreamMuxTilerBintr)
        {
            // Link All Tiler Elementrs and add as the next component in the Branch
            m_pStreamMuxTilerBintr->SetBatchSize(m_batchSize);
            if (!m_pStreamMuxTilerBintr->LinkAll() or
                !m_linkedComponents.back()->LinkToSink(m_pStreamMuxTilerBintr))
            {
                return false;
            }
            m_linkedComponents.push_back(m_pStreamMuxTilerBintr);
            LOG_INFO("Pipeline '" << GetName() << "' Linked up Tiler '" << 
                m_pStreamMuxTilerBintr->GetName() 
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
            if (!m_pPipelineSourcesBintr->StreamMuxPlayTypeIsLiveGet())
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
        
        // Try and lock the Display mutex first
        if (!g_mutex_trylock(&m_displayMutex))
        {
            // lock-failed which means we are already in the XWindow thread context
            // calling on a client handler function for Key release or xWindow delete. 
            // Safe to stop the Pipeline in this context.
            LOG_INFO("dsl_pipeline_stop called from XWindow display thread context");
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
            g_mutex_unlock(&m_displayMutex);
            return true;
        }
        
        // If the main loop is running -- normal case -- then we can't change the 
        // state of the Pipeline in the Application's context. 
        if ((m_pMainLoop and g_main_loop_is_running(m_pMainLoop)) or
            (!m_pMainLoop and g_main_loop_is_running(
                DSL::Services::GetServices()->GetMainLoopHandle())))
        {
            LOG_INFO("Sending application message to stop the pipeline");
            
            gst_element_post_message(GetGstElement(),
                gst_message_new_application(GetGstObject(),
                    gst_structure_new_empty("stop-pipline")));
        }            
        // Else, client has stopped the main-loop or we are running under test 
        // without the mainloop running - can't send a message so handle stop now.
        else
        {
            HandleStop();
        }
        g_mutex_unlock(&m_displayMutex);
        g_mutex_unlock(&m_busWatchMutex);
        return true;
    }

    void PipelineBintr::HandleStop()
    {
        LOG_FUNC();
        
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

//            if (!msg or GST_MESSAGE_TYPE(msg) != GST_MESSAGE_EOS)
//            {
                // TODO - need to review why the 'HandleBusWatchMessage' cb
                // is getting the message in some cases.
//                LOG_WARN("Pipeline '" << GetName() 
//                    << "' failed to receive final EOS message on dsl_pipeline_stop");
//            }
        }

        if (!SetState(GST_STATE_NULL, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
        {
            LOG_ERROR("Failed to Stop Pipeline '" << GetName() << "'");
        }
        
        m_eosFlag = false;
        UnlinkAll();
    }

    bool PipelineBintr::IsLive()
    {
        LOG_FUNC();
        
        if (!m_pPipelineSourcesBintr)
        {
            LOG_INFO("Pipeline '" << GetName() << "' has no sources, therefore is-live = false");
            return false;
        }
        return m_pPipelineSourcesBintr->StreamMuxPlayTypeIsLiveGet();
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
