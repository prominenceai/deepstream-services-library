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
#include "DslPlayerBintr.h"

namespace DSL
{
    PlayerBintr::PlayerBintr(const char* name, 
        DSL_FILE_SOURCE_PTR pSource, DSL_SINK_PTR pSink)
        : Bintr(name, true) // Pipeline = true
        , PipelineBusMgr(m_pGstObj)
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
        
    }

    PlayerBintr::~PlayerBintr()
    {
        LOG_FUNC();
        Stop();
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
    
    bool PlayerBintr::Stop()
    {
        LOG_FUNC();

        if (!IsLinked())
        {
            LOG_WARN("PipelineBase '" << GetName() << "' is not in a state of Playing");
            return false;
        }

        if (!SetState(GST_STATE_NULL, DSL_DEFAULT_STATE_CHANGE_TIMEOUT_IN_SEC * GST_SECOND))
        {
            LOG_ERROR("Failed to Stop Pipeline '" << GetName() << "'");
        }
        UnlinkAll();
        return true;
    }
    
} // DSL   