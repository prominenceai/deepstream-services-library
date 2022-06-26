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
#include "DslTilerBintr.h"
#include "DslBranchBintr.h"

namespace DSL
{

    TilerBintr::TilerBintr(const char* name, uint width, uint height)
        : Bintr(name)
        , m_rows(0)
        , m_columns(0)
        , m_width(width)
        , m_height(height)
        , m_showSourceId(-1)
        , m_showSourceTimeout(0)
        , m_showSourceCounter(0)
        , m_showSourceTimerId(0)
        , m_showSourceCycle(false)
    {
        LOG_FUNC();

        m_pQueue = DSL_ELEMENT_NEW("queue", name);
        m_pTiler = DSL_ELEMENT_NEW("nvmultistreamtiler", name);

        // Don't overwrite the default "best-fit" columns and rows on construction
        m_pTiler->SetAttribute("width", m_width);
        m_pTiler->SetAttribute("height", m_height);
        m_pTiler->SetAttribute("gpu-id", m_gpuId);
        m_pTiler->SetAttribute("nvbuf-memory-type", m_nvbufMemType);

        AddChild(m_pQueue);
        AddChild(m_pTiler);

        m_pQueue->AddGhostPadToParent("sink");
        m_pTiler->AddGhostPadToParent("src");
    
        m_pSinkPadProbe = DSL_PAD_BUFFER_PROBE_NEW("tiler-sink-pad-probe", "sink", m_pQueue);
        m_pSrcPadProbe = DSL_PAD_BUFFER_PROBE_NEW("tiler-src-pad-probe", "src", m_pTiler);
    
        g_mutex_init(&m_showSourceMutex);
    }

    TilerBintr::~TilerBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
        if (m_showSourceTimerId)
        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_showSourceMutex);
            
            g_source_remove(m_showSourceTimerId);
        }
        g_mutex_clear(&m_showSourceMutex);
    }

    bool TilerBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' tiler to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddTilerBintr(shared_from_this());
    }
    
    bool TilerBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("TilerBintr '" << m_name << "' is already linked");
            return false;
        }
        m_pQueue->LinkToSink(m_pTiler);
        m_isLinked = true;
        
        return true;
    }
    
    void TilerBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("TilerBintr '" << m_name << "' is not linked");
            return;
        }
        m_pQueue->UnlinkFromSink();
        m_isLinked = false;
    }
    
    void TilerBintr::GetTiles(uint* columns, uint* rows)
    {
        LOG_FUNC();
        
        m_pTiler->GetAttribute("columns", &m_columns);
        m_pTiler->GetAttribute("rows", &m_rows);
        
        *columns = m_columns;
        *rows = m_rows;
    }
    
    bool TilerBintr::SetTiles(uint columns, uint rows)
    {
        LOG_FUNC();
        
        m_columns = columns;
        m_rows = rows;
    
        m_pTiler->SetAttribute("columns", columns);
        m_pTiler->SetAttribute("rows", m_rows);
        
        return true;
    }
    
    void TilerBintr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        m_pTiler->GetAttribute("width", &m_width);
        m_pTiler->GetAttribute("height", &m_height);
        
        *width = m_width;
        *height = m_height;
    }

    bool TilerBintr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();

        m_width = width;
        m_height = height;

        m_pTiler->SetAttribute("width", m_width);
        m_pTiler->SetAttribute("height", m_height);
        
        return true;
    }

    void TilerBintr::GetShowSource(int* sourceId, uint* timeout)
    {
        LOG_FUNC();
        
        *sourceId = m_showSourceId;
        *timeout = m_showSourceTimeout;
    }

    bool TilerBintr::SetShowSource(int sourceId, uint timeout, bool hasPrecedence)
    {
        // Not logging function entry/exit as this serice can be called by actions for
        // every object in every frame.
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_showSourceMutex);

        // Note: batch size is 0 until Tiler is linked... calls will fail until then
        if (sourceId < 0 or sourceId >= (int)m_batchSize)
        {
            LOG_ERROR("Invalid source Id '" << sourceId << "' for TilerBintr '" << GetName());
            return false;
        }

        // call has Precendence over source cycling 
        m_showSourceCycle = false;
        if (sourceId != m_showSourceId)
        {
            if (m_showSourceTimerId and !hasPrecedence)
            {
                // don't log error as this may be common with ODE Triggers and Actions calling
                LOG_DEBUG("Show source Timer is running for Source '" << m_showSourceId << 
                   "' New Source '" << sourceId << "' without precedence can not be shown");
                return false;
            }

            m_showSourceId = sourceId;
            m_showSourceTimeout = timeout;
            m_pTiler->SetAttribute("show-source", m_showSourceId);

            m_showSourceCounter = m_showSourceTimeout*10;
            if (m_showSourceCounter)
            {
                LOG_INFO("Adding show-source timer with timeout = " << timeout << "' for TilerBintr '" << GetName());
                m_showSourceTimerId = g_timeout_add(100, ShowSourceTimerHandler, this);
            }
            return true;
        }
            
        // otherwise it's the same source.
        
        m_showSourceTimeout = timeout;
        m_showSourceCounter = timeout*10;
        if (!m_showSourceTimerId and m_showSourceCounter)
        {
            LOG_INFO("Adding show-source timer with timeout = " << timeout << "' for TilerBintr '" << GetName());
            m_showSourceTimerId = g_timeout_add(100, ShowSourceTimerHandler, this);
        }
        return true;
    }
    
    bool TilerBintr::CycleAllSources(uint timeout)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_showSourceMutex);
        
        if (!timeout)
        {
            LOG_ERROR("Timeout value can not be 0 when enabling cycle-all-sources for TilerBintr '" << GetName());
            return false;
        }
        // if the timer is currently running, stop and remove first.
        if (m_showSourceTimerId)
        {
            g_source_remove(m_showSourceTimerId);
            m_showSourceTimerId = 0;
        }

        m_showSourceCycle = true;
        m_showSourceId = 0;
        m_showSourceTimeout = timeout;
        m_pTiler->SetAttribute("show-source", m_showSourceId);

        m_showSourceCounter = m_showSourceTimeout*10;
            
        if (!m_showSourceTimerId and m_showSourceCounter)
        {
            LOG_INFO("Adding show-source timer with timeout = " << timeout << "' for TilerBintr '" << GetName());
            m_showSourceTimerId = g_timeout_add(100, ShowSourceTimerHandler, this);
        }
        return true;
    }
    
    
    void TilerBintr::ShowAllSources()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_showSourceMutex);
        
        if (m_showSourceTimerId)
        {
            g_source_remove(m_showSourceTimerId);
            m_showSourceTimerId = 0;
            // call has Precendence over source cycling 
            m_showSourceCycle = false;
        }
        if (m_showSourceId != -1)
        {
            m_showSourceId = -1;
            m_pTiler->SetAttribute("show-source", m_showSourceId);
        }
    }

    int TilerBintr::HandleShowSourceTimer()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_showSourceMutex);
        
        // Tiler is no longer linked but the main_loop and timer are still running
        if (!IsLinked())
        {
            // do nothing, but keep the timer running to support relink and play
            // The Timer's Cycle Source setting should remain as is.
            return true;
        }
        if (--m_showSourceCounter == 0)
        {
            // if we are cycling through sources
            if (m_showSourceCycle)
            {
                // reset the timeout counter, cycle to the next source, and return true to continue
                m_showSourceCounter = m_showSourceTimeout*10;
                m_showSourceId = (m_showSourceId+1)%m_batchSize;
                m_pTiler->SetAttribute("show-source", m_showSourceId);
                return true;
            }
            // otherwise, reset the timer Id, show all sources, and return false to destroy the timer
            m_showSourceTimerId = 0;
            m_showSourceId = -1;
            m_pTiler->SetAttribute("show-source", m_showSourceId);
            return false;
        }
        return true;
    }

    bool TilerBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set GPU ID for TilerBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_gpuId = gpuId;
        LOG_DEBUG("Setting GPU ID to '" << gpuId << "' for TilerBintr '" << m_name << "'");

        m_pTiler->SetAttribute("gpu-id", m_gpuId);
        
        return true;
    }

    bool TilerBintr::SetNvbufMemType(uint nvbufMemType)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Unable to set NVIDIA buffer memory type for TilerBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_nvbufMemType = nvbufMemType;
        m_pTiler->SetAttribute("nvbuf-memory-type", m_nvbufMemType);

        return true;
    }

    //----------------------------------------------------------------------------------------------
    
    static int ShowSourceTimerHandler(void* user_data)
    {
        return static_cast<TilerBintr*>(user_data)->
            HandleShowSourceTimer();
    }
}