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
#include "DslTrackerBintr.h"
#include "DslBranchBintr.h"

namespace DSL
{
    TrackerBintr::TrackerBintr(const char* name, 
        const char* llLibFileName, guint width, guint height)
        : Bintr(name)
        , m_llLibFile(llLibFileName)
        , m_width(width)
        , m_height(height)
    {
        LOG_FUNC();
        m_pTracker = DSL_ELEMENT_NEW(NVDS_ELEM_TRACKER, "tracker-tracker");

        m_pTracker->SetAttribute("tracker-width", m_width);
        m_pTracker->SetAttribute("tracker-height", m_height);
        m_pTracker->SetAttribute("gpu-id", m_gpuId);
        m_pTracker->SetAttribute("ll-lib-file", llLibFileName);
        m_pTracker->SetAttribute("enable-batch-process", true);

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
        
        // add 'this' display to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddTrackerBintr(shared_from_this());
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
    
    const char* TrackerBintr::GetConfigFile()
    {
        LOG_FUNC();
        
        return m_llConfigFile.c_str();
    }
    
    void TrackerBintr::GetMaxDimensions(uint* width, uint* height)
    {
        LOG_FUNC();
        
        m_pTracker->GetAttribute("tracker-width", &m_width);
        m_pTracker->GetAttribute("tracker-height", &m_height);
        
        *width = m_width;
        *height = m_height;
    }

    bool TrackerBintr::SetMaxDimensions(uint width, uint height)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Tiles for TrackerBintr '" << GetName() 
                << "' as it's currently in use");
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

    KtlTrackerBintr::KtlTrackerBintr(const char* name, guint width, guint height)
        : TrackerBintr(name, NVDS_KLT_LIB, width, height)
    {
        LOG_FUNC();
    }
    
    IouTrackerBintr::IouTrackerBintr(const char* name, const char* configFile, guint width, guint height)
        : TrackerBintr(name, NVDS_IOU_LIB, width, height)
    {
        LOG_FUNC();

        m_llConfigFile = configFile;

        std::ifstream streamConfigFile(configFile);
        if (!streamConfigFile.good())
        {
            LOG_ERROR("IOU Tracker Config File '" << configFile << "' Not found");
            throw;
        }
        m_pTracker->SetAttribute("ll-config-file", configFile);
    }
} // DSL