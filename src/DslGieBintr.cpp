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
    PrimaryGieBintr::PrimaryGieBintr(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, uint interval, uint uniqueId)
        : Bintr(name)
        , m_batchSize(0)
        , m_interval(interval)
        , m_uniqueId(uniqueId)
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
        
        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "primary_gie_queue");
        m_pVidConv = DSL_ELEMENT_NEW(NVDS_ELEM_VIDEO_CONV, "primary_gie_conv");
        m_pClassifier = DSL_ELEMENT_NEW(NVDS_ELEM_PGIE, "primary_gie_classifier");

        m_pVidConv->SetAttribute("gpu-id", m_gpuId);
        m_pVidConv->SetAttribute("nvbuf-memory-type", m_nvbufMemoryType);

        m_pClassifier->SetAttribute("config-file-path", inferConfigFile);
        m_pClassifier->SetAttribute("process-mode", 1);
        m_pClassifier->SetAttribute("interval", m_interval);
        m_pClassifier->SetAttribute("unique-id", m_uniqueId);
        m_pClassifier->SetAttribute("gpu-id", m_gpuId);
        m_pClassifier->SetAttribute("model-engine-file", modelEngineFile);
        
        AddChild(m_pQueue);
        AddChild(m_pVidConv);
        AddChild(m_pClassifier);

        m_pQueue->AddGhostPadToParent("sink");
        m_pClassifier->AddGhostPadToParent("src");
    }    
    
    PrimaryGieBintr::~PrimaryGieBintr()
    {
        LOG_FUNC();

        UnlinkAll();
    }

    bool PrimaryGieBintr::LinkAll()
    {
        LOG_FUNC();
        
        m_pQueue->LinkToSink(m_pVidConv);
        m_pVidConv->LinkToSink(m_pClassifier);
        
        m_isLinked = true;
        
        return true;
    }
    
    void PrimaryGieBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        m_pQueue->UnlinkFromSink();
        m_pVidConv->UnlinkFromSink();

        m_isLinked = false;
    }

    void PrimaryGieBintr::AddToParent(DSL_NODETR_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' GIE to the Parent Pipeline 
        std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            AddPrimaryGieBintr(shared_from_this());
    }

    const char* PrimaryGieBintr::GetInferConfigFile()
    {
        LOG_FUNC();
        
        return m_inferConfigFile.c_str();
    }
    
    const char* PrimaryGieBintr::GetModelEngineFile()
    {
        LOG_FUNC();
        
        return m_modelEngineFile.c_str();
    }
    
    void PrimaryGieBintr::SetBatchSize(uint batchSize)
    {
        LOG_FUNC();
        
        m_batchSize = batchSize;
        m_pClassifier->SetAttribute("batch-size", m_batchSize);
    }
    
    uint PrimaryGieBintr::GetBatchSize()
    {
        LOG_FUNC();
        
        return m_batchSize;
    }

    void PrimaryGieBintr::SetInterval(uint interval)
    {
        LOG_FUNC();
        
        m_interval = interval;
        m_pClassifier->SetAttribute("interval", m_interval);
    }
    
    uint PrimaryGieBintr::GetInterval()
    {
        LOG_FUNC();
        
        return m_interval;
    }

    void PrimaryGieBintr::SetUniqueId(uint uniqueId)
    {
        LOG_FUNC();
        
        m_uniqueId = uniqueId;
        m_pClassifier->SetAttribute("unique-id", m_uniqueId);
    }
    
    uint PrimaryGieBintr::GetUniqueId()
    {
        LOG_FUNC();
        
        return m_uniqueId;
    }
    
}    
