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
        m_uniqueId = (uint)std::hash<std::string>{}(name);
        
        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "gie_queue");
        m_pClassifier = DSL_ELEMENT_NEW(factoryname, "gie_classifier");

        m_pClassifier->SetAttribute("config-file-path", inferConfigFile);
        m_pClassifier->SetAttribute("process-mode", m_processMode);
        m_pClassifier->SetAttribute("interval", m_interval);
        m_pClassifier->SetAttribute("unique-id", m_uniqueId);
        m_pClassifier->SetAttribute("gpu-id", m_gpuId);
        m_pClassifier->SetAttribute("model-engine-file", modelEngineFile);
        
        AddChild(m_pQueue);
        AddChild(m_pClassifier);

        m_pQueue->AddGhostPadToParent("sink");
        m_pClassifier->AddGhostPadToParent("src");
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
        m_pClassifier->SetAttribute("batch-size", m_batchSize);
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
        m_pClassifier->SetAttribute("interval", m_interval);
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
    
    PrimaryGieBintr::PrimaryGieBintr(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, uint interval)
        : GieBintr(name, NVDS_ELEM_PGIE, 1, inferConfigFile, modelEngineFile, interval)
    {
        LOG_FUNC();
        
        m_pVidConv = DSL_ELEMENT_NEW(NVDS_ELEM_VIDEO_CONV, "primary_gie_conv");

        m_pVidConv->SetAttribute("gpu-id", m_gpuId);
        m_pVidConv->SetAttribute("nvbuf-memory-type", m_nvbufMemoryType);

        AddChild(m_pVidConv);
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
        m_pQueue->LinkToSink(m_pVidConv);
        m_pVidConv->LinkToSink(m_pClassifier);
        
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

    SecondaryGieBintr::SecondaryGieBintr(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, uint interval, const char* inferOnGieName)
        : GieBintr(name, NVDS_ELEM_SGIE, 2, inferConfigFile, modelEngineFile, interval)
        , m_inferOnGieName(inferOnGieName)
    {
        LOG_FUNC();
        
        std::size_t inferOnGieId = std::hash<std::string>{}(inferOnGieName);
        m_pClassifier->SetAttribute("infer-on-gie-id", (guint)inferOnGieId);
        
        m_pSink = DSL_ELEMENT_NEW(NVDS_ELEM_SINK_FAKESINK, "secondary-gie-sink");
        m_pSink->SetAttribute("async", false);
        m_pSink->SetAttribute("sync", false);
        m_pSink->SetAttribute("enable-last-sample", false);
        
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
            LOG_ERROR("PrimaryGieBintr '" << m_name << "' is already linked");
            return false;
        }
        m_pQueue->LinkToSink(m_pClassifier);
        m_pClassifier->LinkToSink(m_pSink);
        
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
        m_pClassifier->UnlinkFromSink();
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

        m_pGstRequestedSourcePads["src"] = pGstRequestedSourcePad;

        return Bintr::LinkToSource(pTee);
    }

    bool SecondaryGieBintr::UnlinkFromSource()
    {
        LOG_FUNC();
        
        // If we're not currently linked to the Tee
        if (!IsLinkedToSource())
        {
            LOG_ERROR("SecondaryGieBintr '" << GetName() << "' is not in a Linked state");
            return false;
        }

        LOG_INFO("Unlinking and releasing requested Source Pad for SecondaryGieBintr " << GetName());
        
        gst_pad_unlink(m_pGstRequestedSourcePads["src"], m_pGstStaticSinkPad);
        gst_element_release_request_pad(GetSource()->GetGstElement(), m_pGstRequestedSourcePads["src"]);
                
        m_pGstRequestedSourcePads.erase("src");
        
        return Nodetr::UnlinkFromSource();
    }
}    
