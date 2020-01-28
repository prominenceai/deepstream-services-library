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
#include "DslDewarperBintr.h"
#include "DslPipelineBintr.h"

namespace DSL
{

    DewarperBintr::DewarperBintr(const char* name, const char* configFile)
        : Bintr(name)
        , m_configFile(configFile)
    {
        LOG_FUNC();

        m_configFile = configFile;

        std::ifstream streamConfigFile(configFile);
        if (!streamConfigFile.good())
        {
            LOG_ERROR("Dewarper Config File '" << configFile << "' Not found");
            throw;
        }
        
        m_pSinkQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "dewarper-sink-queue");
        m_pVidConv = DSL_ELEMENT_NEW(NVDS_ELEM_VIDEO_CONV, "dewarper-vid-conv");
        m_pVidCaps = DSL_ELEMENT_NEW(NVDS_ELEM_CAPS_FILTER, "dewarper-vid-caps");
        m_pDewarper = DSL_ELEMENT_NEW(NVDS_ELEM_DEWARPER, "dewarper");
        m_pDewarperCaps = DSL_ELEMENT_NEW(NVDS_ELEM_CAPS_FILTER, "dewarper-caps");
        m_pSrcQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "dewarper-src-queue");

        m_pVidConv->SetAttribute("gpu-id", m_gpuId);
        m_pVidConv->SetAttribute("nvbuf-memory-type", m_nvbufMemoryType);

        // Set Capabilities filter for Video Converter 
        GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGBA", NULL);
        gst_caps_set_features(caps, 0, gst_caps_features_new(MEMORY_FEATURES, NULL));
        m_pVidCaps->SetAttribute("caps", caps);
        gst_caps_unref(caps);

        m_pDewarper->SetAttribute("gpu-id", m_gpuId);
        m_pDewarper->SetAttribute("config-file", configFile);

        // Set Capabilities filter for Dewarper
        caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING,
            "RGBA", "width", GST_TYPE_INT_RANGE, 1, G_MAXINT,
            "height", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL);
        gst_caps_set_features(caps, 0, gst_caps_features_new("memory:NVMM", NULL));
        m_pDewarperCaps->SetAttribute("caps", caps);
        gst_caps_unref(caps);


        AddChild(m_pSinkQueue);
        AddChild(m_pVidConv);
        AddChild(m_pVidCaps);
        AddChild(m_pDewarper);
        AddChild(m_pDewarperCaps);
        AddChild(m_pSrcQueue);

        m_pSinkQueue->AddGhostPadToParent("sink");
        m_pSrcQueue->AddGhostPadToParent("src");
    }

    DewarperBintr::~DewarperBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }

    bool DewarperBintr::AddToParent(DSL_NODETR_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' display to the Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            AddDewarperBintr(shared_from_this());
    }
    
    bool DewarperBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("DewarperBintr '" << m_name << "' is already linked");
            return false;
        }
        m_pSinkQueue->LinkToSink(m_pVidConv);
        m_pVidConv->LinkToSink(m_pVidCaps);
        m_pVidCaps->LinkToSink(m_pDewarper);
        m_pDewarper->LinkToSink(m_pDewarperCaps);
        m_pDewarperCaps->LinkToSink(m_pSrcQueue);
        
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
        m_pSinkQueue->UnlinkFromSink();
        m_pVidConv->UnlinkFromSink();
        m_pVidCaps->UnlinkFromSink();
        m_pDewarper->UnlinkFromSink();
        m_pDewarperCaps->UnlinkFromSink();
        
        m_isLinked = false;
    }

    const char* DewarperBintr::GetConfigFile()
    {
        LOG_FUNC();
        
        return m_configFile.c_str();
    }
    
    bool  DewarperBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set GPU ID for DewarperBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_gpuId = gpuId;
        LOG_DEBUG("Setting GPU ID to '" << gpuId << "' for DewarperBintr '" << m_name << "'");

        m_pVidConv->SetAttribute("gpu-id", m_gpuId);
        m_pDewarper->SetAttribute("gpu-id", m_gpuId);
        return true;
    }

}