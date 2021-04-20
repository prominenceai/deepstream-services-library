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
#include "DslServices.h"
#include "DslPlayerBintr.h"

namespace DSL
{
    PlayerBintr::PlayerBintr(const char* name, 
        DSL_SOURCE_PTR pSource, DSL_SINK_PTR pSink)
        : Bintr(name, true) // Pipeline = true
        , PipelineStateMgr(m_pGstObj)
        , PipelineXWinMgr(m_pGstObj)
        , m_pSource(pSource)
        , m_pSink(pSink)
    {
        LOG_FUNC();

        g_mutex_init(&m_asyncCommMutex);
        
        if (!AddChild(m_pSource))
        {
            LOG_ERROR("Failed to add SourceBintr '" << m_pSource->GetName() 
                << "' to PlayerBintr '" << GetName() << "'");
            throw;
        }
        if (!AddChild(m_pSink))
        {
            LOG_ERROR("Failed to add SinkBintr '" << m_pSink->GetName() 
                << "' to PlayerBintr '" << GetName() << "'");
            throw;
        }
        
        AddEosListener(PlayerTerminate, this);
        AddXWindowDeleteEventHandler(PlayerTerminate, this);
    }

    PlayerBintr::~PlayerBintr()
    {
        LOG_FUNC();
        Stop();
        RemoveEosListener(PlayerTerminate);
        RemoveXWindowDeleteEventHandler(PlayerTerminate);
        g_mutex_clear(&m_asyncCommMutex);
    }

    bool PlayerBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("PlayerBintr '" << GetName() << "' is already linked");
            return false;
        }
        if (!m_pSource->LinkAll() or ! m_pSink->LinkAll() or 
            !m_pSource->LinkToSink(m_pSink))
        {
            LOG_ERROR("Failed link SourceBintr '" << m_pSource->GetName() 
                << "' to SinkBintr '" << m_pSink->GetName() << "'");
            return false;
        }
        m_isLinked = true;
        return true;
    }

    void PlayerBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("PlayerBintr '" << GetName() << "' is not linked");
            return;
        }
        if (!m_pSource->UnlinkFromSink())
        {
            LOG_ERROR("Failed unlink SourceBintr '" << m_pSource->GetName() 
                << "' to SinkBintr '" << m_pSink->GetName() << "'");
            return;
        }
        m_pSource->UnlinkAll();
        m_pSink->UnlinkAll();
        m_isLinked = false;
    }
    
    bool PlayerBintr::Play()
    {
        LOG_FUNC();

        GstState currentState;
        GetState(currentState, 0);
        if (currentState == GST_STATE_NULL)
        {
            if (!LinkAll())
            {
                LOG_ERROR("Unable to prepare Pipeline '" << GetName() << "' for Play");
                return false;
            }
            if (!SetState(GST_STATE_PAUSED, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
            {
                LOG_ERROR("Failed to Pause before playing Pipeline '" << GetName() << "'");
                return false;
            }

        }
        // Call the base class to complete the Play process
        return SetState(GST_STATE_PLAYING, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND);
    }
    
    bool PlayerBintr::Pause()
    {
        LOG_FUNC();
        
        GstState state;
        GetState(state, 0);
        if (state != GST_STATE_PLAYING)
        {
            LOG_WARN("Player '" << GetName() << "' is not in a state of Playing");
            return false;
        }
        // If the main loop is running -- normal case -- then we can't change the 
        // state of the Pipeline in the Application's context. 
        if (g_main_loop_is_running(DSL::Services::GetServices()->GetMainLoopHandle()))
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);
            g_timeout_add(1, PlayerPause, this);
            g_cond_wait(&m_asyncCondition, &m_asyncCommMutex);
        }
        // Else, we are running under test without the mainloop
        else
        {
            HandlePause();
        }
        return true;
    }

    void PlayerBintr::HandlePause()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);
        
        // Call the base class to Pause
        if (!SetState(GST_STATE_PAUSED, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
        {
            LOG_ERROR("Failed to Pause Player '" << GetName() << "'");
        }
        if (g_main_loop_is_running(DSL::Services::GetServices()->GetMainLoopHandle()))
        {
            g_cond_signal(&m_asyncCondition);
        }
    }

    bool PlayerBintr::Stop()
    {
        LOG_FUNC();
        
        if (!IsLinked())
        {
            return false;
        }

        SendEos();
        sleep(1);

        // If the main loop is running -- normal case -- then we can't change the 
        // state of the Pipeline in the Application's context. 
        if (g_main_loop_is_running(DSL::Services::GetServices()->GetMainLoopHandle()))
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);
            g_timeout_add(1, PlayerStop, this);
            g_cond_wait(&m_asyncCondition, &m_asyncCommMutex);
        }
        // Else, we are running under test without the mainloop
        else
        {
            HandleStop();
        }
        return true;
    }
    
    void PlayerBintr::HandleStop()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);

        if (!SetState(GST_STATE_NULL, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
        {
            LOG_ERROR("Failed to Stop Pipeline '" << GetName() << "'");
        }
        UnlinkAll();
        if (g_main_loop_is_running(DSL::Services::GetServices()->GetMainLoopHandle()))
        {
            g_cond_signal(&m_asyncCondition);
        }
        // iterate through the map of Termination event listeners calling each
        for(auto const& imap: m_terminationEventListeners)
        {
            try
            {
                imap.first(imap.second);
            }
            catch(...)
            {
                LOG_ERROR("Exception calling Client Termination event Listener");
            }
        }
    }

    void PlayerBintr::HandleTermination()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_asyncCommMutex);

        // Start asyn Stop timer, do not wait or block as we are
        // in the State Manager's bus watcher context.
        g_timeout_add(1, PlayerStop, this);
        
        DestroyXWindow();
    }

    
    bool PlayerBintr::AddTerminationEventListener(
        dsl_player_termination_event_listener_cb listener, void* clientData)
    {
        LOG_FUNC();

        if (m_terminationEventListeners.find(listener) != m_terminationEventListeners.end())
        {   
            LOG_ERROR("Player listener is not unique");
            return false;
        }
        m_terminationEventListeners[listener] = clientData;
        
        return true;
    }
    
    bool PlayerBintr::RemoveTerminationEventListener(
        dsl_player_termination_event_listener_cb listener)
    {
        LOG_FUNC();

        if (m_terminationEventListeners.find(listener) == m_terminationEventListeners.end())
        {   
            LOG_ERROR("Player listener was not found");
            return false;
        }
        m_terminationEventListeners.erase(listener);
        
        return true;
    }
    
    static int PlayerPause(gpointer pPlayer)
    {
        static_cast<PlayerBintr*>(pPlayer)->HandlePause();
        
        // Return false to self destroy timer - one shot.
        return false;
    }
    
    static int PlayerStop(gpointer pPlayer)
    {
        static_cast<PlayerBintr*>(pPlayer)->HandleStop();
        
        // Return false to self destroy timer - one shot.
        return false;
    }
    
    static void PlayerTerminate(void* pPlayer)
    {
        static_cast<PlayerBintr*>(pPlayer)->HandleTermination();
    }    
    
} // DSL   