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
        if (state == GST_STATE_PLAYING or state == GST_STATE_PAUSED)
        {
            Stop();
        }
        SetState(GST_STATE_NULL, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND);
        g_mutex_clear(&m_asyncCommMutex);
    }
    
    bool PipelineBintr::AddSourceBintr(DSL_BASE_PTR pSourceBintr)
    {
        LOG_FUNC();

        if (!m_pPipelineSourcesBintr->AddChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr)))
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
        return (m_pPipelineSourcesBintr->IsChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr)));
    }

    bool PipelineBintr::RemoveSourceBintr(DSL_BASE_PTR pSourceBintr)
    {
        LOG_FUNC();

        // Must cast to SourceBintr first so that correct Instance of RemoveChild is called
        return m_pPipelineSourcesBintr->RemoveChild(std::dynamic_pointer_cast<SourceBintr>(pSourceBintr));
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
            LOG_ERROR("Pipeline '" << GetName() << "' is currently Linked - cudadec memory type can not be updated");
            return false;
            
        }
        m_pPipelineSourcesBintr->SetStreamMuxNvbufMemType(type);
        
        return true;
    }

    void PipelineBintr::GetStreamMuxBatchProperties(guint* batchSize, uint* batchTimeout)
    {
        LOG_FUNC();

        *batchSize = m_batchSize;
        *batchTimeout = m_batchTimeout;
    }

    bool PipelineBintr::SetStreamMuxBatchProperties(uint batchSize, uint batchTimeout)
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("Pipeline '" << GetName() << "' is currently Linked - batch properties can not be updated");
            return false;
            
        }
        m_batchSize = batchSize;
        m_batchTimeout = batchTimeout;
        m_pPipelineSourcesBintr->SetStreamMuxBatchProperties(m_batchSize, m_batchTimeout);
        
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

        // If the batch size has not been explicitely set, use the number of sources.
        if (m_batchSize < m_pPipelineSourcesBintr->GetNumChildren())
        {
            SetStreamMuxBatchProperties(m_pPipelineSourcesBintr->GetNumChildren(), m_batchTimeout);
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
        // If the main loop is running -- normal case -- then we can't change the 
        // state of the Pipeline in the Application's context. 
        if (g_main_loop_is_running(DSL::Services::GetServices()->GetMainLoopHandle()))
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);
            g_timeout_add(1, PipelinePause, this);
            g_cond_wait(&m_asyncCondition, &m_asyncCommMutex);
        }
        // Else, we are running under test without the mainloop
        else
        {
            HandlePause();
        }
        return true;
    }

    void PipelineBintr::HandlePause()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);
        
        // Call the base class to Pause
        if (!SetState(GST_STATE_PAUSED, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
        {
            LOG_ERROR("Failed to Pause Pipeline '" << GetName() << "'");
        }
        if (g_main_loop_is_running(DSL::Services::GetServices()->GetMainLoopHandle()))
        {
            g_cond_signal(&m_asyncCondition);
        }
    }

    bool PipelineBintr::Stop()
    {
        LOG_FUNC();
        
        if (!IsLinked())
        {
            return false;
        }
        // If the main loop is running -- normal case -- then we can't change the 
        // state of the Pipeline in the Application's context. 
        if (g_main_loop_is_running(DSL::Services::GetServices()->GetMainLoopHandle()))
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);
            g_timeout_add(1, PipelineStop, this);
            g_cond_wait(&m_asyncCondition, &m_asyncCommMutex);
        }
        // Else, we are running under test without the mainloop
        else
        {
            HandleStop();
        }
        return true;
    }

    void PipelineBintr::HandleStop()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);

        // Call on all sources to disable their EOS consumers, before send EOS
        m_pPipelineSourcesBintr->DisableEosConsumers();
        
        SendEos();
        sleep(1);

        if (!SetState(GST_STATE_READY, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
        {
            LOG_ERROR("Failed to Stop Pipeline '" << GetName() << "'");
        }
        UnlinkAll();
        if (g_main_loop_is_running(DSL::Services::GetServices()->GetMainLoopHandle()))
        {
            g_cond_signal(&m_asyncCondition);
        }
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

    static int PipelinePause(gpointer pPipeline)
    {
        static_cast<PipelineBintr*>(pPipeline)->HandlePause();
        
        // Return false to self destroy timer - one shot.
        return false;
    }
    
    static int PipelineStop(gpointer pPipeline)
    {
        static_cast<PipelineBintr*>(pPipeline)->HandleStop();
        
        // Return false to self destroy timer - one shot.
        return false;
    }

} // DSL
