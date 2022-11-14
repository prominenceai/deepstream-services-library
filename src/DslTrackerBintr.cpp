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
#include "DslTrackerBintr.h"
#include "DslBranchBintr.h"

namespace DSL
{
    TrackerBintr::TrackerBintr(const char* name,
        const char* configFile, guint width, guint height)
        : Bintr(name)
        , m_llLibFile(NVDS_MOT_LIB)
        , m_llConfigFile(configFile)
        , m_width(width)
        , m_height(height)
        , m_batchProcessingEnabled(true)
        , m_pastFrameReporting(false)
    {
        LOG_FUNC();
        
        // ktl
        m_pTracker = DSL_ELEMENT_NEW("nvtracker", name);

        m_pTracker->SetAttribute("tracker-width", m_width);
        m_pTracker->SetAttribute("tracker-height", m_height);
        m_pTracker->SetAttribute("gpu-id", m_gpuId);
        m_pTracker->SetAttribute("ll-lib-file", m_llLibFile.c_str());

        // set the low-level configuration file property if provided.
        if (m_llConfigFile.size())
        {
            m_pTracker->SetAttribute("ll-config-file", configFile);
        }

        AddChild(m_pTracker);

        m_pTracker->AddGhostPadToParent("sink");
        m_pTracker->AddGhostPadToParent("src");
        
        m_pSinkPadProbe = DSL_PAD_BUFFER_PROBE_NEW("tracker-sink-pad-probe", "sink", m_pTracker);
        m_pSrcPadProbe = DSL_PAD_BUFFER_PROBE_NEW("tracker-src-pad-probe", "src", m_pTracker);
    }

    TrackerBintr::~TrackerBintr()
    {
        LOG_FUNC();

        if (IsLinked())
        {
            UnlinkAll();
        }
    }

    bool TrackerBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' Tracker to the Parent Branch 
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddTrackerBintr(shared_from_this());
    }

    bool TrackerBintr::RemoveFromParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // remove 'this' Tracker from the Parent Branch
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            RemoveTrackerBintr(shared_from_this());
    }
    
    bool TrackerBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("TrackerBintr '" << m_name << "' is already linked");
            return false;
        }
        // Nothing to link with single Elementr
        m_isLinked = true;
        
        return true;
    }
    
    void TrackerBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("TrackerBintr '" << m_name << "' is not linked");
            return;
        }
        // Nothing to unlink with single Elementr
        m_isLinked = false;
    }

    const char* TrackerBintr::GetLibFile()
    {
        LOG_FUNC();
        
        return m_llLibFile.c_str();
    }
    
    bool TrackerBintr::SetLibFile(const char* libFile)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set library file for TrackerBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        m_llLibFile.assign(libFile);
        m_pTracker->SetAttribute("ll-lib-file", libFile);
        return true;
    }
    
    const char* TrackerBintr::GetConfigFile()
    {
        LOG_FUNC();
        
        return m_llConfigFile.c_str();
    }
    
    bool TrackerBintr::SetConfigFile(const char* configFile)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set config file for TrackerBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        m_llConfigFile.assign(configFile);
        m_pTracker->SetAttribute("ll-config-file", configFile);
        return true;
    }
    
    void TrackerBintr::GetDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        m_pTracker->GetAttribute("tracker-width", &m_width);
        m_pTracker->GetAttribute("tracker-height", &m_height);
        
        *width = m_width;
        *height = m_height;
    }

    bool TrackerBintr::SetDimensions(uint width, uint height)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Dimensions for TrackerBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }

        m_width = width;
        m_height = height;

        m_pTracker->SetAttribute("tracker-width", m_width);
        m_pTracker->SetAttribute("tracker-height", m_height);
        
        return true;
    }

    bool TrackerBintr::SetGpuId(uint gpuId)
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

        m_pTracker->SetAttribute("gpu-id", m_gpuId);
        
        return true;
    }

    bool TrackerBintr::GetBatchProcessingEnabled()
    {
        LOG_FUNC();

        return m_batchProcessingEnabled;
    }
    
    bool TrackerBintr::SetBatchProcessingEnabled(bool enabled)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set the enable-batch-processing setting for TrackerBintr '" 
                << GetName() << "' as it's currently in use");
            return false;
        }
        
        m_batchProcessingEnabled = enabled;
        m_pTracker->SetAttribute("enable-batch-process", m_batchProcessingEnabled);
        return true;
    }
    
    bool TrackerBintr::GetPastFrameReportingEnabled()
    {
        LOG_FUNC();

        return m_pastFrameReporting;
    }
    
    bool TrackerBintr::SetPastFrameReportingEnabled(bool enabled)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set the enable-past-frame setting for TrackerBintr '" 
                << GetName() << "' as it's currently in use");
            return false;
        }
        m_pastFrameReporting = enabled;
        m_pTracker->SetAttribute("enable-past-frame", m_pastFrameReporting);
        return true;
    }

    bool TrackerBintr::SetBatchSize(uint batchSize)
    {
        LOG_FUNC();
        
        if (batchSize > 1 and !m_batchProcessingEnabled)
        {
            LOG_WARN("The Pipeline's batch-size is set to " << batchSize 
                << " while the Tracker's batch processing is disable!");
        }
        return true;
    }
    
} // DSL