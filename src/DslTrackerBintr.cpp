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
#include "DslTrackerBintr.h"
#include "DslBranchBintr.h"

namespace DSL
{
    TrackerBintr::TrackerBintr(const char* name,
        const char* configFile, guint width, guint height)
        : QBintr(name)
        , m_llLibFile(NVDS_MOT_LIB)
        , m_llConfigFile(configFile)
        , m_width(width)
        , m_height(height)
        , m_trackOnGieId(0)
    {
        LOG_FUNC();

        // New Queue and Tracker element for this TrackerBintr
        m_pTracker = DSL_ELEMENT_NEW("nvtracker", name);

        m_pTracker->SetAttribute("tracker-width", m_width);
        m_pTracker->SetAttribute("tracker-height", m_height);
        m_pTracker->SetAttribute("ll-lib-file", m_llLibFile.c_str());

        // set the low-level configuration file property if provided.
        if (m_llConfigFile.size())
        {
            m_pTracker->SetAttribute("ll-config-file", configFile);
        }

        // Get property defaults that aren't specifically set
        m_pTracker->GetAttribute("input-tensor-meta", &m_tensorInputEnabled);
        m_pTracker->GetAttribute("display-tracking-id", &m_idDisplayEnabled);
        m_pTracker->GetAttribute("gpu-id", &m_gpuId);

        LOG_INFO("");
        LOG_INFO("Initial property values for TrackerBintr '" << name << "'");
        LOG_INFO("  tracker-width        : " << m_width);
        LOG_INFO("  tracker-height       : " << m_height);
        LOG_INFO("  ll-lib-file          : " << m_llLibFile);
        LOG_INFO("  ll-config-file       : " << m_llConfigFile);
        LOG_INFO("  display-tracking-id  : " << m_idDisplayEnabled);
        LOG_INFO("  input-tensor-meta    : " << m_tensorInputEnabled);
        LOG_INFO("  gpu-id               : " << m_gpuId);
        LOG_INFO("  queue                : " );
        LOG_INFO("    leaky              : " << m_leaky);
        LOG_INFO("    max-size           : ");
        LOG_INFO("      buffers          : " << m_maxSizeBuffers);
        LOG_INFO("      bytes            : " << m_maxSizeBytes);
        LOG_INFO("      time             : " << m_maxSizeTime);
        LOG_INFO("    min-threshold      : ");
        LOG_INFO("      buffers          : " << m_minThresholdBuffers);
        LOG_INFO("      bytes            : " << m_minThresholdBytes);
        LOG_INFO("      time             : " << m_minThresholdTime);

        AddChild(m_pTracker);

        // Float the queue element as a sink-ghost-pad for this Bintr.
        m_pQueue->AddGhostPadToParent("sink");

        // Float the tracker element as a src-ghost-pad for this Bintr.
        m_pTracker->AddGhostPadToParent("src");
        
        // Add the Buffer and DS Event probes to the tracker element.
        AddSinkPadProbes(m_pTracker->GetGstElement());
        AddSrcPadProbes(m_pTracker->GetGstElement());
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

        if (!m_pQueue->LinkToSink(m_pTracker))
        {
            return false;
        }
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
        m_pQueue->UnlinkFromSink();

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
            LOG_ERROR("Unable to set GPU ID for TrackerBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_gpuId = gpuId;
        m_pTracker->SetAttribute("gpu-id", m_gpuId);
        
        LOG_INFO("TrackerBintr '" << GetName() 
            << "' - new GPU ID = " << m_gpuId );

        return true;
    }

    void TrackerBintr::GetTensorMetaSettings(boolean* inputEnabled, 
        const char** trackOnGie)
    {
        LOG_FUNC();

        *inputEnabled = m_tensorInputEnabled;
        *trackOnGie = m_trackOnGieName.c_str();
    }
    
    bool TrackerBintr::SetTensorMetaSettings(boolean inputEnabled, 
        const char* trackOnGie)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set the tensor-meta-settings for TrackerBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_tensorInputEnabled = inputEnabled;
        m_trackOnGieName = trackOnGie;
        
        m_pTracker->SetAttribute("input-tensor-meta", m_tensorInputEnabled);

        if (m_tensorInputEnabled)
        {
            // Query the Services lib for the Id of the GIE to track on. 
            
            uint inferOnProcessMode; // do we care about process mode????
            
            if (Services::GetServices()->_inferAttributesGetByName(
                m_trackOnGieName.c_str(), m_trackOnGieId, inferOnProcessMode) 
                    != DSL_RESULT_SUCCESS)
            {
                LOG_ERROR("TrackerBintr '" << GetName() 
                    << "' failed to Get unique Id for InferBintr '" 
                    << m_trackOnGieName << "'");
                return false;
            }   
            
            LOG_INFO("Setting tensor-meta-gie-id for TrackerBintr   '" 
                << GetName() << "' to " << m_trackOnGieId);
                
            m_pTracker->SetAttribute("tensor-meta-gie-id", m_trackOnGieId);
        }    
        return true;
    }
    
    boolean TrackerBintr::GetIdDisplayEnabled()
    {
        LOG_FUNC();

        m_pTracker->GetAttribute("display-tracking-id", &m_idDisplayEnabled);
        return m_idDisplayEnabled;
    }
    
    bool TrackerBintr::SetIdDisplayEnabled(boolean enabled)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set the display-tracking-id for TrackerBintr '" 
                << GetName() << "' as it's currently in linked");
            return false;
        }
        m_idDisplayEnabled = enabled;
        m_pTracker->SetAttribute("display-tracking-id", m_idDisplayEnabled);
        return true;
    }
    
} // DSL