/*
The MIT License

Copyright (c) 2019-Present, ROBERT HOWELL

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
        , m_showSourceCounter(0)
        , m_showSourceTimerId(0)    
    {
        LOG_FUNC();

        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "tiler-queue");
        m_pTiler = DSL_ELEMENT_NEW(NVDS_ELEM_TILER, "tiler-tiler");

        // Don't overwrite the default "best-fit" columns and rows on construction
        m_pTiler->SetAttribute("width", m_width);
        m_pTiler->SetAttribute("height", m_height);
        m_pTiler->SetAttribute("gpu-id", m_gpuId);
        m_pTiler->SetAttribute("nvbuf-memory-type", m_nvbufMemoryType);

        AddChild(m_pQueue);
        AddChild(m_pTiler);

        m_pQueue->AddGhostPadToParent("sink");
        m_pTiler->AddGhostPadToParent("src");
    
        m_pSinkPadProbe = DSL_PAD_PROBE_NEW("tiler-sink-pad-probe", "sink", m_pQueue);
        m_pSrcPadProbe = DSL_PAD_PROBE_NEW("tiler-src-pad-probe", "src", m_pTiler);
    
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
    
    void TilerBintr::GetTiles(uint* rows, uint* columns)
    {
        LOG_FUNC();
        
        *rows = m_rows;
        *columns = m_columns;
    }
    
    bool TilerBintr::SetTiles(uint rows, uint columns)
    {
        LOG_FUNC();
        
        m_rows = rows;
        m_columns = columns;
    
        m_pTiler->SetAttribute("rows", m_rows);
        m_pTiler->SetAttribute("columns", m_rows);
        
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
        *timeout = m_showSourceCounter;
    }

    bool TilerBintr::SetShowSource(int sourceId, uint timeout)
    {
        // Don't log function entry/exit as this could be called frequently
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_showSourceMutex);

        if (sourceId < 0 or sourceId >= (int)m_batchSize)
        {
            LOG_ERROR("Invalid source Id '" << sourceId << "' for TilerBintr '" << GetName());
            return false;
        }
        
        if (sourceId != m_showSourceId)
        {
            if (m_showSourceTimerId)
            {
                // don't log error as this may be common with ODE Triggers and Actions calling
                LOG_INFO("Show source Timer is running for Source '" << m_showSourceId << 
                   "' New Source '" << sourceId << "' can not be shown");
                return false;
            }

            m_showSourceId = sourceId;
            m_pTiler->SetAttribute("show-source", m_showSourceId);

            m_showSourceCounter = timeout*10;
            if (m_showSourceCounter)
            {
                LOG_INFO("Adding show-source timer with timeout = " << timeout << "' for TilerBintr '" << GetName());
                m_showSourceTimerId = g_timeout_add(100, ShowSourceTimerHandler, this);
            }
            return true;
        }
            
        // otherwise it's the same source.
        
        m_showSourceCounter = timeout*10;
        if (!m_showSourceTimerId and m_showSourceCounter)
        {
            LOG_INFO("Adding show-source timer with timeout = " << timeout << "' for TilerBintr '" << GetName());
            m_showSourceTimerId = g_timeout_add(100, ShowSourceTimerHandler, this);
        }
        return true;
    }
    
    int TilerBintr::HandleShowSourceTimer()
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_showSourceMutex);
        
        if (--m_showSourceCounter == 0)
        {
            // reset the timer Id, show all sources, and return false to destroy the timer
            m_showSourceTimerId = 0;
            m_showSourceId = -1;
            m_pTiler->SetAttribute("show-source", m_showSourceId);
            return false;
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
        }
        if (m_showSourceId != -1)
        {
            m_showSourceId = -1;
            m_pTiler->SetAttribute("show-source", m_showSourceId);
        }
    }

    bool TilerBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set GPU ID for FileSinkBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_gpuId = gpuId;
        LOG_DEBUG("Setting GPU ID to '" << gpuId << "' for FileSinkBintr '" << m_name << "'");

        m_pTiler->SetAttribute("gpu-id", m_gpuId);
        
        return true;
    }

    //----------------------------------------------------------------------------------------------
    
    static int ShowSourceTimerHandler(void* user_data)
    {
        return static_cast<TilerBintr*>(user_data)->
            HandleShowSourceTimer();
    }
}