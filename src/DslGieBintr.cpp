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
#include "DslBranchBintr.h"

namespace DSL
{
    GieBintr::GieBintr(const char* name, const char* factoryname, uint processMode,
        const char* inferConfigFile, const char* modelEngineFile)
        : Bintr(name)
        , m_processMode(processMode)
        , m_interval(0)
        , m_uniqueId(CreateUniqueIdFromName(name))
        , m_inferConfigFile(inferConfigFile)
        , m_modelEngineFile(modelEngineFile)
        , m_rawOutputEnabled(false)
        , m_rawOutputFrameNumber(0)
    {
        LOG_FUNC();
        
        std::ifstream streamInferConfigFile(inferConfigFile);
        if (!streamInferConfigFile.good())
        {
            LOG_ERROR("Infer Config File '" << inferConfigFile << "' Not found");
            throw;
        }        
        // generate a unique Id for the GIE based on its unique name
        std::string gieName = "gie-" + GetName();
        
        LOG_INFO("Creating GIE  '" << gieName << "' with unique Id = " << m_uniqueId);
        
        // create and setup unique GIE Elementr
        m_pInferEngine = DSL_ELEMENT_NEW(factoryname, gieName.c_str());
        m_pInferEngine->SetAttribute("config-file-path", inferConfigFile);
        m_pInferEngine->SetAttribute("process-mode", m_processMode);
        m_pInferEngine->SetAttribute("unique-id", m_uniqueId);
        m_pInferEngine->SetAttribute("gpu-id", m_gpuId);
        m_pInferEngine->SetAttribute("model-engine-file", modelEngineFile);
        
        g_object_set (m_pInferEngine->GetGstObject(),
            "raw-output-generated-callback", OnRawOutputGeneratedCB,
            "raw-output-generated-userdata", this,
            NULL);
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

    bool GieBintr::SetInferConfigFile(const char* inferConfigFile)
    {
        LOG_FUNC();
        
        std::ifstream streamInferConfigFile(inferConfigFile);
        if (!streamInferConfigFile.good())
        {
            LOG_ERROR("Infer Config File '" << inferConfigFile << "' Not found");
            return false;
        }        
                
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Infer Config File for GIE '" << GetName() 
                << "' as it's currently in use");
            return false;
        }
        m_inferConfigFile.assign(inferConfigFile);
        m_pInferEngine->SetAttribute("config-file-path", inferConfigFile);
        
        return true;
    }
    
    const char* GieBintr::GetModelEngineFile()
    {
        LOG_FUNC();
        
        return m_modelEngineFile.c_str();
    }
    
    bool GieBintr::SetModelEngineFile(const char* modelEngineFile)
    {
        LOG_FUNC();
        
        std::ifstream streamModelEngineFile(modelEngineFile);
        if (!streamModelEngineFile.good())
        {
            LOG_ERROR("Model Engine File '" << modelEngineFile << "' Not found");
            return false;
        }        
                
        if (IsInUse())
        {
            LOG_ERROR("Unable to set Model Engine File for GIE '" << GetName() 
                << "' as it's currently in use");
            return false;
        }
        m_modelEngineFile.assign(modelEngineFile);
        m_pInferEngine->SetAttribute("model-engine-file", modelEngineFile);
        
        return true;
    }
    
    bool GieBintr::SetBatchSize(uint batchSize)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Batch size for GIE '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        m_pInferEngine->SetAttribute("batch-size", batchSize);
        return Bintr::SetBatchSize(batchSize);
    }

    bool GieBintr::SetInterval(uint interval)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Interval for GIE '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        m_interval = interval;
        m_pInferEngine->SetAttribute("interval", m_interval);
        
        return true;
    }
    
    uint GieBintr::GetInterval()
    {
        LOG_FUNC();
        
        return m_interval;
    }

    int GieBintr::GetUniqueId()
    {
        LOG_FUNC();
        
        return m_uniqueId;
    }
    
    int GieBintr::CreateUniqueIdFromName(const char* name)
    {
        LOG_FUNC();
        
        // TODO this is a temporary work-around for the fact that the
        // Infer Engine "unique-id" is unsigned (guit), but the "infer-on-gie-id"
        // is a signed (gint), and fails on negative values. Need to use a static 
        // hash table, or try and get the parameter type changed ?? or some means 
        // of eliminating the possible collision that can come from below
        
        // generate a unique Id based on name, and ensure positive signed integer
        int id = std::hash<std::string>{}(name);
        return (id < 0) ? id*-1 : id;
    }

    bool GieBintr::SetRawOutputEnabled(bool enabled, const char* path)
    {
        LOG_FUNC();
        
        if (enabled)
        {
            struct stat info;

            if( stat(path, &info) != 0 )
            {
                LOG_ERROR("Unable to access path '" << path << "' for GieBintr '" << GetName() << "'");
                return false;
            }
            else if(info.st_mode & S_IFDIR)
            {
                LOG_INFO("Enabling raw layer-info output to path '" << path << "' for GieBintr '" << GetName() << "'");
                m_rawOutputPath.assign(path);
            }
            else
            {
                LOG_ERROR("Unable to access path '" << path << "' for GieBintr '" << GetName() << "'");
                return false;
            }
        }
        else
        {
            LOG_INFO("Disabling raw layer-info output to path '" << m_rawOutputPath << "' for GieBintr '" << GetName() << "'");
            m_rawOutputPath.clear();
        }
        m_rawOutputEnabled = enabled;
        return true;
    }

    void GieBintr::HandleOnRawOutputGeneratedCB(GstBuffer* pBuffer, NvDsInferNetworkInfo* pNetworkInfo, 
        NvDsInferLayerInfo *pLayersInfo, guint layersCount, guint batchSize)
    {
        if (!m_rawOutputEnabled)
        {
            return;
        }
        for (int i=0; i<layersCount; i++)
        {
            NvDsInferLayerInfo *pLayerInfo = &pLayersInfo[i];
            
            std::string layerName(pLayerInfo->layerName);
            std::replace(layerName.begin(), layerName.end(), '/', '_');
            std::string oFilePath = m_rawOutputPath + "/" + layerName + 
                "_batch" + std::to_string(m_rawOutputFrameNumber) + "_bsize" + std::to_string(batchSize) + ".bin";
                
            std::ofstream streamOutputFile(oFilePath, std::ofstream::out | std::ofstream::binary);
            if (!streamOutputFile.good())
            {
                LOG_ERROR("Failed to open '" << oFilePath << "' - check path");
                return;
            }        
            uint typeSize;
            switch (pLayerInfo->dataType) {
                case FLOAT: typeSize = 4; break;
                case HALF: typeSize = 2; break;
                case INT32: typeSize = 4; break;
                case INT8: typeSize = 1; break;
            }
            streamOutputFile.write((char*)pLayerInfo->buffer, typeSize * pLayerInfo->dims.numElements * batchSize);
            streamOutputFile.close();
        }
        m_rawOutputFrameNumber++;
    }

    static void OnRawOutputGeneratedCB(GstBuffer* pBuffer, NvDsInferNetworkInfo* pNetworkInfo, 
        NvDsInferLayerInfo *pLayersInfo, guint layersCount, guint batchSize, gpointer pGie)
    {
        static_cast<GieBintr*>(pGie)->HandleOnRawOutputGeneratedCB(pBuffer, pNetworkInfo, 
            pLayersInfo, layersCount, batchSize);
    }

    // ***********************************************************************
    
    PrimaryGieBintr::PrimaryGieBintr(const char* name, const char* inferConfigFile,
        const char* modelEngineFile, uint interval)
        : GieBintr(name, NVDS_ELEM_PGIE, 1, inferConfigFile, modelEngineFile)
    {
        LOG_FUNC();
        
        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, "primary-gie-queue");
        m_pVidConv = DSL_ELEMENT_NEW(NVDS_ELEM_VIDEO_CONV, "primary-gie-conv");

        m_pVidConv->SetAttribute("gpu-id", m_gpuId);
        m_pVidConv->SetAttribute("nvbuf-memory-type", m_nvbufMemoryType);
        
        // update the InferEngine interval setting
        SetInterval(interval);

        AddChild(m_pQueue);
        AddChild(m_pVidConv);
        AddChild(m_pInferEngine);

        m_pQueue->AddGhostPadToParent("sink");
        m_pInferEngine->AddGhostPadToParent("src");
        
        m_pSinkPadProbe = DSL_PAD_PROBE_NEW("gie-sink-pad-probe", "sink", m_pQueue);
        m_pSrcPadProbe = DSL_PAD_PROBE_NEW("gie-src-pad-probe", "src", m_pInferEngine);
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

        if (!m_batchSize)
        {
            LOG_ERROR("PrimaryGieBintr '" << GetName() << "' can not be linked: batch size = 0");
            return false;
        }
        if (m_isLinked)
        {
            LOG_ERROR("PrimaryGieBintr '" << GetName() << "' is already linked");
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
            LOG_ERROR("PrimaryGieBintr '" << GetName() << "' is not linked");
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
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddPrimaryGieBintr(shared_from_this());
    }

    bool PrimaryGieBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set GPU ID for Primary GIE '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_gpuId = gpuId;
        LOG_DEBUG("Setting GPU ID to '" << gpuId << "' for PrimaryBintr '" << m_name << "'");

        m_pInferEngine->SetAttribute("gpu-id", m_gpuId);
        m_pVidConv->SetAttribute("gpu-id", m_gpuId);
        
        return true;
    }

    // ***********************************************************************

    SecondaryGieBintr::SecondaryGieBintr(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, const char* inferOnGieName, uint interval)
        : GieBintr(name, NVDS_ELEM_SGIE, 2, inferConfigFile, modelEngineFile)
    {
        LOG_FUNC();
        
        // create the unique queue-name from the SGIE name
        std::string queueName = "sgie-queue-" + GetName();

        m_pQueue = DSL_ELEMENT_NEW(NVDS_ELEM_QUEUE, queueName.c_str());

        
        // update the InferEngine interval setting
        SetInferOnGieName(inferOnGieName);
        SetInterval(interval);
        
        // create the unique sink-name from the SGIE name
        std::string fakeSinkName = "sgie-fake-sink-" + GetName();
        
        m_pFakeSink = DSL_ELEMENT_NEW(NVDS_ELEM_SINK_FAKESINK, fakeSinkName.c_str());
        m_pFakeSink->SetAttribute("async", false);
        m_pFakeSink->SetAttribute("sync", false);
        m_pFakeSink->SetAttribute("enable-last-sample", false);
        
        // Note: the Elementrs created/owned by this SecondaryGieBintr are added as 
        // children to the parent PipelineSGiesBintr, and not to this Bintr's GST BIN
        // In this way, all Secondary GIEs Infer on the same buffer of data, regardless
        // of the depth of secondary Inference. Ghost Pads are not required for this bin
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

        if (!m_batchSize)
        {
            LOG_ERROR("SecondaryGieBintr '" << GetName() << "' can not be linked: batch size = 0");
            return false;
        }
        if (m_isLinked)
        {
            LOG_ERROR("SecondaryGieBintr '" << GetName() << "' is already linked");
            return false;
        }
        if (!m_pQueue->LinkToSink(m_pInferEngine) or !m_pInferEngine->LinkToSink(m_pFakeSink))
        {
            LOG_ERROR("SecondaryGieBintr '" << GetName() << "' failed to link");
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
            LOG_ERROR("SecondaryGieBintr '" << GetName() << "' is not linked");
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
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
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
    
    bool SecondaryGieBintr::SetInferOnGieName(const char* name)
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("Unable to update SecondaryGieBintr '" << GetName() << 
                "' as its in a linked state");
            return false;
            
        }
        LOG_WARN(name);
        m_inferOnGieName.assign(name);
        m_inferOnGieUniqueId = CreateUniqueIdFromName(name);
        
        LOG_INFO("Setting infer-on-gie-id for SecondaryGieBintr '" << GetName() << "' to " << m_inferOnGieUniqueId);
        
        m_pInferEngine->SetAttribute("infer-on-gie-id", m_inferOnGieUniqueId);
        
        return true;
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

    bool SecondaryGieBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set GPU ID for Secondary GIE '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_gpuId = gpuId;
        LOG_DEBUG("Setting GPU ID to '" << gpuId << "' for DewarperBintr '" << m_name << "'");

        m_pInferEngine->SetAttribute("gpu-id", m_gpuId);
        return true;
    }
}    
