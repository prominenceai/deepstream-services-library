/*
The MIT License

Copyright (c) 2019-2023, Prominence AI, Inc.

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
#include "DslDewarperBintr.h"
#include "DslPipelineBintr.h"

namespace DSL
{

    DewarperBintr::DewarperBintr(const char* name, 
        const char* configFile, uint cameraId)
        : Bintr(name)
        , m_configFile(configFile)
        , m_cameraId(cameraId)
    {
        LOG_FUNC();

        m_pDewarper = DSL_ELEMENT_NEW("nvdewarper", name);

    
        m_pDewarper->SetAttribute("config-file", configFile);
        m_pDewarper->SetAttribute("source-id", m_cameraId);
        
        // Get properties not explicitly set
        m_pDewarper->GetAttribute("num-batch-buffers", &m_numBatchBuffers);

        LOG_INFO("");
        LOG_INFO("Initial property values for AppSourceBintr '" << name << "'");
        LOG_INFO("  config-file       : " << m_configFile);
        LOG_INFO("  camera-id         : " << m_cameraId);
        LOG_INFO("  gpu-id            : " << m_gpuId);
        LOG_INFO("  num-batch-buffers : " << m_numBatchBuffers);
        LOG_INFO("  nvbuf-memory-type : " << m_nvbufMemType);

        AddChild(m_pDewarper);

        m_pDewarper->AddGhostPadToParent("sink");
        m_pDewarper->AddGhostPadToParent("src");
    }

    DewarperBintr::~DewarperBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }

    bool DewarperBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // Dewarper should not be added to Pipeline 
        // Must add to source directy
        LOG_ERROR("DewarperBintr '" << m_name 
            << "' can not be added to a Pipeline directly. Add to Source");
        return false;
    }
    
    bool DewarperBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("DewarperBintr '" << m_name << "' is already linked");
            return false;
        }
        
        // single element - nothing to link
        m_isLinked = true;
        
        return true;
    }
    
    void DewarperBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("DewarperBintr '" << m_name << "' is not linked");
            return;
        }
        // single element - nothing to link
        
        m_isLinked = false;
    }

    const char* DewarperBintr::GetConfigFile()
    {
        LOG_FUNC();
        
        return m_configFile.c_str();
    }

    bool DewarperBintr::SetConfigFile(const char* configFile)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set config-file for DewarperBintr '" << GetName() 
                << "' as it's currently in a linked state");
            return false;
        }
        
        m_configFile = configFile;
        m_pDewarper->SetAttribute("config-file", configFile);
        
        return true;
    }

    uint DewarperBintr::GetCameraId()
    {
        LOG_FUNC();
        
        return m_cameraId;
    }
    
    bool DewarperBintr::SetCameraId(uint cameraId)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set camera-id for DewarperBintr '" << GetName() 
                << "' as it's currently in a linked state");
            return false;
        }

        m_cameraId = cameraId;

        m_pDewarper->SetAttribute("source-id", m_cameraId);
        return true;
    }
    
    bool  DewarperBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set gpu-id for DewarperBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_gpuId = gpuId;
        LOG_DEBUG("Setting GPU ID to '" << m_gpuId 
            << "' for DewarperBintr '" << m_name << "'");

        m_pDewarper->SetAttribute("gpu-id", m_gpuId);
        return true;
    }

    uint DewarperBintr::GetNumBatchBuffers()
    {
        LOG_FUNC();
        
        return m_numBatchBuffers;
    }

    bool DewarperBintr::SetNumBatchBuffers(uint num)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Unable to set num-batch-buffers for DewarperBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }

        m_numBatchBuffers = num;

        m_pDewarper->SetAttribute("num-batch-buffers", m_numBatchBuffers);
        return true;
    }
    
    bool DewarperBintr::SetNvbufMemType(uint nvbufMemType)
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("Unable to set nvbuf-memory-type for DewarperBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_nvbufMemType = nvbufMemType;
        m_pDewarper->SetAttribute("nvbuf-memory-type", m_nvbufMemType);

        return true;
    }

}