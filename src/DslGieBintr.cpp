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
#include "DslGieBintr.h"
#include "DslPipelineBintr.h"

namespace DSL
{
    GieBintr::GieBintr(const char* name, const char* factoryname, uint processMode,
        const char* inferConfigFile, const char* modelEngineFile, uint interval)
        : Bintr(name)
        , m_processMode(processMode)
        , m_batchSize(0)
        , m_interval(interval)
        , m_inferConfigFile(inferConfigFile)
        , m_modelEngineFile(modelEngineFile)
    {
        LOG_FUNC();
        
        std::ifstream streamInferConfigFile(inferConfigFile);
        if (!streamInferConfigFile.good())
        {
            LOG_ERROR("Infer Config File '" << inferConfigFile << "' Not found");
            throw;
        }        

        // generate a unique Id for the GIE based on its unique name
        m_uniqueId = std::hash<std::string>{}(name);
        
        // unique element name 
        std::string gieName = "gie-" + GetName();
        
        LOG_INFO("Creating GIE  '" << gieName << "' with unique Id = " << m_uniqueId);
        
        // create and setup unique GIE Elementr
        m_pInferEngine = DSL_ELEMENT_NEW(factoryname, gieName.c_str());

        m_pInferEngine->SetAttribute("config-file-path", inferConfigFile);
        m_pInferEngine->SetAttribute("process-mode", m_processMode);
        m_pInferEngine->SetAttribute("interval", m_interval);
        m_pInferEngine->SetAttribute("unique-id", m_uniqueId);
        m_pInferEngine->SetAttribute("gpu-id", m_gpuId);
        m_pInferEngine->SetAttribute("model-engine-file", modelEngineFile);
    }    
    
    GieBintr::~GieBintr()
    {
        LOG_FUNC();
    }

    const char* GieBintr::GetInferConfigFile()
    {
        LOG_FUNC();
        
        return m_inferConfigFile.c_str();
    }
    
    const char* GieBintr::GetModelEngineFile()
    {
        LOG_FUNC();
        
        return m_modelEngineFile.c_str();
    }
    
    void GieBintr::SetBatchSize(uint batchSize)
    {
        LOG_FUNC();
        
        m_batchSize = batchSize;
        m_pInferEngine->SetAttribute("batch-size", m_batchSize);
    }
    
    uint GieBintr::GetBatchSize()
    {
        LOG_FUNC();
        
        return m_batchSize;
    }

    void GieBintr::SetInterval(uint interval)
    {
        LOG_FUNC();
        
        m_interval = interval;
        m_pInferEngine->SetAttribute("interval", m_interval);
    }
    
    uint GieBintr::GetInterval()
    {
        LOG_FUNC();
        
        return m_interval;
    }

    uint GieBintr::GetUniqueId()
    {
        LOG_FUNC();
        
        return m_uniqueId;
    }

    // ***********************************************************************
    // ***********************************************************************
    
    PrimaryGieBintr::PrimaryGieBintr(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, uint interval)
        : GieBintr(name, NVDS_ELEM_PGIE, 1, inferConfigFile, modelEngineFile, interval)
    {
        LOG_FUNC();
        
        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "primary-gie-queue");
        m_pVidConv = DSL_ELEMENT_NEW(NVDS_ELEM_VIDEO_CONV, "primary-gie-conv");

        m_pVidConv->SetAttribute("gpu-id", m_gpuId);
        m_pVidConv->SetAttribute("nvbuf-memory-type", m_nvbufMemoryType);

        AddChild(m_pInferEngine);
        AddChild(m_pVidConv);
        AddChild(m_pQueue);

        m_pQueue->AddGhostPadToParent("sink");
        m_pInferEngine->AddGhostPadToParent("src");
    }    
    
    PrimaryGieBintr::~PrimaryGieBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }

    bool PrimaryGieBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("PrimaryGieBintr '" << m_name << "' is already linked");
            return false;
        }
        if (!m_pQueue->LinkToSink(m_pVidConv) or !m_pVidConv->LinkToSink(m_pInferEngine))
        {
            return false;
        }
        
        m_isLinked = true;
        
        return true;
    }
    
    void PrimaryGieBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("PrimaryGieBintr '" << m_name << "' is not linked");
            return;
        }
        m_pQueue->UnlinkFromSink();
        m_pVidConv->UnlinkFromSink();

        m_isLinked = false;
    }

    bool PrimaryGieBintr::AddToParent(DSL_NODETR_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' GIE to the Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            AddPrimaryGieBintr(shared_from_this());
    }

    // ***********************************************************************
    // ***********************************************************************

    SecondaryGieBintr::SecondaryGieBintr(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, uint interval, const char* inferOnGieName)
        : GieBintr(name, NVDS_ELEM_SGIE, 2, inferConfigFile, modelEngineFile, interval)
        , m_inferOnGieName(inferOnGieName)
    {
        LOG_FUNC();
        
        m_inferOnGieUniqueId = std::hash<std::string>{}(inferOnGieName);
        m_pInferEngine->SetAttribute("infer-on-gie-id", m_inferOnGieUniqueId);
        
        // create the unique queue-name from the SGIE name
        std::string queueName = "sgie-queue-" + GetName();

        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, queueName.c_str());
        
        // create the unique sink-name from the SGIE name
        std::string fakeSinkName = "sgie-fake-sink-" + GetName();
        
        m_pFakeSink = DSL_ELEMENT_NEW(NVDS_ELEM_SINK_FAKESINK, fakeSinkName.c_str());
        m_pFakeSink->SetAttribute("async", false);
        m_pFakeSink->SetAttribute("sync", false);
        m_pFakeSink->SetAttribute("enable-last-sample", false);
    }    
    
    SecondaryGieBintr::~SecondaryGieBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }

    bool SecondaryGieBintr::LinkAll()
    {
        LOG_FUNC();

        if (m_isLinked)
        {
            LOG_ERROR("SecondaryGieBintr '" << m_name << "' is already linked");
            return false;
        }
        if (!m_pQueue->LinkToSink(m_pInferEngine) or !m_pInferEngine->LinkToSink(m_pFakeSink))
        {
            LOG_ERROR("SecondaryGieBintr '" << m_name << "' failed to link");
            return false;
        }
        
        m_isLinked = true;
        
        return true;
    }
    
    void SecondaryGieBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("SecondaryGieBintr '" << m_name << "' is not linked");
            return;
        }
        m_pInferEngine->UnlinkFromSink();
        m_pQueue->UnlinkFromSink();

        m_isLinked = false;
    }

    bool SecondaryGieBintr::AddToParent(DSL_NODETR_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' GIE to the Parent Pipeline 
        return std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            AddSecondaryGieBintr(shared_from_this());
    }
    
    uint SecondaryGieBintr::GetInferOnGieUniqueId()
    {
        LOG_FUNC();

        return m_inferOnGieUniqueId;
    }
    
    const char* SecondaryGieBintr::GetInferOnGieName()
    {
        LOG_FUNC();

        return m_inferOnGieName.c_str();
    }
    
    void SecondaryGieBintr::SetInferOnGieName(const char* name)
    {
        LOG_FUNC();
        
        m_inferOnGieName.assign(name);
    }

    bool SecondaryGieBintr::LinkToSource(DSL_NODETR_PTR pTee)
    {
        LOG_FUNC();

        LOG_INFO("Linking SecondaryGieBintr '" << GetName() << "' to Tee '" << pTee->GetName() << "'");
        
        m_pGstStaticSinkPad = gst_element_get_static_pad(m_pQueue->GetGstElement(), "sink");
        if (!m_pGstStaticSinkPad)
        {
            LOG_ERROR("Failed to get Static Sink Pad for SecondaryGieBintr '" << GetName() << "'");
            return false;
        }

        GstPad* pGstRequestedSourcePad = gst_element_get_request_pad(pTee->GetGstElement(), "src_%u");
            
        if (!pGstRequestedSourcePad)
        {
            LOG_ERROR("Failed to get Tee Pad for  '" << pTee->GetName() <<" '");
            return false;
        }

        if (!m_pQueue->LinkToSource(pTee))
        {
            LOG_ERROR("Failed to link Tee '" << pTee->GetName() << "' with SecondaryGieBintr '" << GetName() << "'");
            return false;
        }

        m_pGstRequestedSourcePads["src"] = pGstRequestedSourcePad;
        
        return true;
    }

    bool SecondaryGieBintr::UnlinkFromSource()
    {
        LOG_FUNC();
        
        // If we're not currently linked to the Tee
        if (!m_pQueue->IsLinkedToSource())
        {
            LOG_ERROR("SecondaryGieBintr '" << GetName() << "' is not in a Linked state");
            return false;
        }

        if (!m_pQueue->UnlinkFromSource())
        {
            LOG_ERROR("SecondaryGieBintr '" << GetName() << "' was not able to unlink from src Tee");
            return false;
        }
        LOG_INFO("Unlinking and releasing requested Source Pad for SecondaryGieBintr " << GetName());
        
//        gst_element_release_request_pad(GetSource()->GetGstElement(), m_pGstRequestedSourcePads["src"]);
                
        m_pGstRequestedSourcePads.erase("src");
        
        return true;
    }
}    
