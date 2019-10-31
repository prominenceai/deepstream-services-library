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
    GieBintr::GieBintr(const char* osd, const char* inferConfigFile,
        guint batchSize, guint interval, guint uniqueId, guint gpuId, 
        const char* modelEngineFile, const char*  rawOutputDir)
        : Bintr(osd)
        , m_batchSize(batchSize)
        , m_interval(interval)
        , m_uniqueId(uniqueId)
    {
        LOG_FUNC();
        
        m_inferConfigFile = inferConfigFile;
        m_modelEngineFile = modelEngineFile;
        m_rawOutputDir = rawOutputDir;
        
        
        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "primary_gie_queue", m_pBin);
        m_pVidConv = DSL_ELEMENT_NEW(NVDS_ELEM_VIDEO_CONV, "primary_gie_conv", m_pBin);
        m_pClassifier = DSL_ELEMENT_NEW(NVDS_ELEM_PGIE, "primary_gie_classifier", m_pBin);

        g_object_set(G_OBJECT(m_pVidConv->m_pElement), 
            "gpu-id", m_gpuId,
            "nvbuf-memory-type", m_nvbufMemoryType, NULL);

        g_object_set(G_OBJECT(m_pClassifier->m_pElement), "config-file-path", 
            (gchar*)inferConfigFile, "process-mode", 1, NULL);

        if (m_batchSize)
        {
            g_object_set(G_OBJECT(m_pClassifier->m_pElement), "batch-size", m_batchSize, NULL);
        }
        
        if (m_interval)
        {
            g_object_set(G_OBJECT(m_pClassifier->m_pElement), "interval", m_interval, NULL);
        }

        if (m_uniqueId)
        {
            g_object_set(G_OBJECT(m_pClassifier->m_pElement), "unique-id", m_uniqueId, NULL);
        }
        
        if (m_gpuId)
        {
            g_object_set(G_OBJECT(m_pClassifier->m_pElement), "gpu-id", m_uniqueId, NULL);
        }

        if (m_modelEngineFile.length())
        {
            g_object_set(G_OBJECT(m_pClassifier->m_pElement), "model-engine-file",
                (gchar*)m_modelEngineFile.c_str(), NULL);
        }

        if (m_rawOutputDir.length())
        {
//            g_object_set(G_OBJECT(m_pClassifier),
//                "raw-output-generated-callback", out_callback,
//                "raw-output-generated-userdata", config,
//                NULL);
        }

        m_pQueue->AddSinkGhostPad();
        m_pClassifier->AddSourceGhostPad();
    }    
    
    GieBintr::~GieBintr()
    {
        LOG_FUNC();
    }

    void GieBintr::LinkAll()
    {
        LOG_FUNC();
        
        m_pQueue->LinkTo(m_pVidConv);
        m_pVidConv->LinkTo(m_pClassifier);
    }
    
    void GieBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        m_pQueue->Unlink();
        m_pVidConv->Unlink();
    }

    void GieBintr::AddToParent(std::shared_ptr<Bintr> pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' GIE to the Parent Pipeline 
        std::dynamic_pointer_cast<PipelineBintr>(pParentBintr)->
            AddPrimaryGieBintr(shared_from_this());
    }
}    