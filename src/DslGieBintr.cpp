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
#include "DslPipeline.h"

namespace DSL
{
    GieBintr::GieBintr(const char* osd, const char* configFilePath,
        guint batchSize, guint interval, guint uniqueId, guint gpuId, 
        const char* modelEngineFile, const char*  rawOutputDir)
        : Bintr(osd)
        , m_batchSize(batchSize)
        , m_interval(interval)
        , m_uniqueId(uniqueId)
        , m_pVidConv(NULL)
        , m_pQueue(NULL)
        , m_pClassifier(NULL)
    {
        LOG_FUNC();
        
        m_configFilePath = configFilePath;
        m_modelEngineFile = modelEngineFile;
        m_rawOutputDir = rawOutputDir;
        
        
        m_pQueue = MakeElement(NVDS_ELEM_QUEUE, "primary_gie_queue", LINK_TRUE);
        m_pVidConv = MakeElement(NVDS_ELEM_VIDEO_CONV, "primary_gie_conv", LINK_TRUE);
        m_pClassifier = MakeElement(NVDS_ELEM_PGIE, "primary_gie_classifier", LINK_TRUE);

        g_object_set(G_OBJECT(m_pVidConv), "gpu-id", m_gpuId, NULL);
        g_object_set(G_OBJECT(m_pVidConv), "nvbuf-memory-type", m_nvbufMemoryType, NULL);

        g_object_set(G_OBJECT(m_pClassifier), "config-file-path", 
            GET_FILE_PATH ((gchar*)configFilePath), "process-mode", 1, NULL);

        if (m_batchSize)
        {
            g_object_set(G_OBJECT(m_pClassifier), "batch-size", m_batchSize, NULL);
        }
        
        if (m_interval)
        {
            g_object_set(G_OBJECT(m_pClassifier), "interval", m_interval, NULL);
        }

        if (m_uniqueId)
        {
            g_object_set(G_OBJECT(m_pClassifier), "unique-id", m_uniqueId, NULL);
        }
        
        if (m_gpuId)
        {
            g_object_set(G_OBJECT(m_pClassifier), "gpu-id", m_uniqueId, NULL);
        }

        if (m_modelEngineFile.length())
        {
            g_object_set(G_OBJECT(m_pClassifier), "model-engine-file",
                (gchar*)m_modelEngineFile.c_str(), NULL);
//                GET_FILE_PATH((gchar*)m_modelEngineFile), NULL);
        }

        if (m_rawOutputDir.length())
        {
//            g_object_set(G_OBJECT(m_pClassifier),
//                "raw-output-generated-callback", out_callback,
//                "raw-output-generated-userdata", config,
//                NULL);
        }


        AddGhostPads();
    }    
    
    GieBintr::~GieBintr()
    {
        LOG_FUNC();
    }

    void GieBintr::AddToPipeline(std::shared_ptr<Pipeline> pPipeline)
    {
        LOG_FUNC();
        
        // add 'this' GIE to the Parent Pipeline 
        pPipeline->AddPrimaryGieBintr(shared_from_this());
    }
}    