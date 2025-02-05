/*
The MIT License

Copyright (c) 2019-2025, Prominence AI, Inc.

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
#include "DslInferBintr.h"
#include "DslBranchBintr.h"

namespace DSL
{
    // Initilize the unique id list for all InferBintrs 
    std::list<uint> InferBintr::s_uniqueIds;

    InferBintr::InferBintr(const char* name, uint processMode, 
        const char* inferConfigFile, const char* modelEngineFile, 
        uint interval, uint inferType)
        : QBintr(name)
        , m_inferType(inferType)
        , m_processMode(processMode)
        , m_interval(interval)
        , m_batchSizeSetByClient(false)
        , m_inferConfigFile(inferConfigFile)
        , m_modelEngineFile(modelEngineFile)
        , m_rawOutputEnabled(false)
        , m_rawOutputFrameNumber(0)
        , m_inputTensorMetaEnabled(false)
        , m_outputTensorMetaEnabled(false)
    {
        LOG_FUNC();

        // Find the first available unique Id
        m_uniqueId = 1; // must start at 1 not 0
        while(std::find(s_uniqueIds.begin(), s_uniqueIds.end(), m_uniqueId) != s_uniqueIds.end())
        {
            m_uniqueId++;
        }
        s_uniqueIds.push_back(m_uniqueId);
        
        std::ifstream streamInferConfigFile(inferConfigFile);
        if (!streamInferConfigFile.good())
        {
            LOG_ERROR("Infer Config File '" << inferConfigFile << "' Not found");
            throw;
        }        
        // generate a unique Id for the GIE based on its unique name
        std::string inferBintrName = "infer-" + GetName();

        // Provide the unique-id, name, and process mode to the services object.
        if (Services::GetServices()->_inferAttributesSet(m_uniqueId, 
            GetCStrName(), m_processMode) != DSL_RESULT_SUCCESS)
        {
            LOG_ERROR("Inference Component '" << GetName() 
                << "' failed to Set attributes for its unique Id");
            throw;
        }   

        // create and setup unique Inference Element, AIE, GIE, or TIS
        if (m_inferType == DSL_INFER_TYPE_AIE)
        {
            m_pInferEngine = DSL_ELEMENT_NEW("nvinferaudio", inferBintrName.c_str());
        }
        else if (m_inferType == DSL_INFER_TYPE_GIE)
        {
            m_pInferEngine = DSL_ELEMENT_NEW("nvinfer", inferBintrName.c_str());
        }
        else if(m_inferType == DSL_INFER_TYPE_TIS)
        {
            m_pInferEngine = DSL_ELEMENT_NEW("nvinferserver", inferBintrName.c_str());
        }
        else
        {
            LOG_ERROR("Invalid Inference Type for Infer Component '" << name << "'");
            throw std::exception();
        }

        // Update common properties for all infer types
        m_pInferEngine->SetAttribute("config-file-path", inferConfigFile);
        m_pInferEngine->SetAttribute("unique-id", m_uniqueId);

        // If nvinfer or nvinferserver (i.e. Video Inference)
        if (m_inferType == DSL_INFER_TYPE_GIE or m_inferType == DSL_INFER_TYPE_TIS)
        {
            m_pInferEngine->SetAttribute("process-mode", m_processMode);
            m_pInferEngine->SetAttribute("interval", m_interval);
        }
        // If nvinfer or nvinferaudio (i.e. not an Inference Server)
        if (m_inferType == DSL_INFER_TYPE_AIE or m_inferType == DSL_INFER_TYPE_GIE)
        {
            m_pInferEngine->GetAttribute("gpu-id", &m_gpuId);
            
            // If a model engine file is provided (non-server only)
            if (m_modelEngineFile.size())
            {
                m_pInferEngine->SetAttribute("model-engine-file", modelEngineFile);
            }
            // connect the callback to the GIE element's model-updated signal
            g_signal_connect(m_pInferEngine->GetGObject(), "model-updated", 
                G_CALLBACK(OnModelUpdatedCB), this);
        }
        
//        g_object_set (m_pInferEngine->GetGstObject(),
//            "raw-output-generated-callback", OnRawOutputGeneratedCB,
//            "raw-output-generated-userdata", this,
//            NULL);

        // Add the Buffer and DS Event probes to the infer-engine element.
        AddSinkPadProbes(m_pInferEngine->GetGstElement());
        AddSrcPadProbes(m_pInferEngine->GetGstElement());
    }    
    
    InferBintr::~InferBintr()
    {
        LOG_FUNC();
        
        Services::GetServices()->_inferAttributesErase(m_uniqueId);
        s_uniqueIds.remove(m_uniqueId);
    }

    uint InferBintr::GetInferType()
    {
        LOG_FUNC();
        
        return m_inferType;
    }

    const char* InferBintr::GetInferConfigFile()
    {
        LOG_FUNC();
        
        return m_inferConfigFile.c_str();
    }

    bool InferBintr::SetInferConfigFile(const char* inferConfigFile)
    {
        LOG_FUNC();
        
        std::ifstream streamInferConfigFile(inferConfigFile);
        if (!streamInferConfigFile.good())
        {
            LOG_ERROR("Infer Config File '" << inferConfigFile << "' Not found");
            return false;
        }        
        // can be updated in any state        
        m_inferConfigFile.assign(inferConfigFile);
        m_pInferEngine->SetAttribute("config-file-path", inferConfigFile);
        
        return true;
    }
    
    const char* InferBintr::GetModelEngineFile()
    {
        LOG_FUNC();
        
        return m_modelEngineFile.c_str();
    }
    
    bool InferBintr::SetModelEngineFile(const char* modelEngineFile)
    {
        LOG_FUNC();
        
        if (m_inferType == DSL_INFER_TYPE_TIS)
        {
            LOG_ERROR("Cannot set Model Engine File for TIS");
            return false;
        }
        
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
    
    bool InferBintr::SetBatchSizeByClient(uint batchSize)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Batch size for InferBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        if (!batchSize)
        {
            m_batchSizeSetByClient = false;
            return true;
        }
        m_batchSizeSetByClient = true;
        
        m_pInferEngine->SetAttribute("batch-size", batchSize);
        return Bintr::SetBatchSize(batchSize);
    }

    bool InferBintr::SetBatchSize(uint batchSize)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Batch size for InferBintr '" << GetName() 
                << "' as it's currently linked");
        }
        if (m_batchSizeSetByClient)
        {
            LOG_INFO("Batch size for InferBintr '" << GetName() 
                << "' explicitely set by client will not be updated.");
            return true;
        }
        m_pInferEngine->SetAttribute("batch-size", batchSize);
        return Bintr::SetBatchSize(batchSize);
    }
    
    uint InferBintr::GetInterval()
    {
        LOG_FUNC();
        
        return m_interval;
    }

    bool InferBintr::SetInterval(uint interval)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Interval for InferBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        m_interval = interval;
        m_pInferEngine->SetAttribute("interval", m_interval);
        
        return true;
    }
    
    int InferBintr::GetUniqueId()
    {
        LOG_FUNC();
        
        return m_uniqueId;
    }
    
    void InferBintr::GetTensorMetaSettings(bool* inputEnabled, bool* outputEnabled)
    {
        LOG_FUNC();
    
        *inputEnabled = m_inputTensorMetaEnabled;
        *outputEnabled = m_outputTensorMetaEnabled;
    }    

    bool InferBintr::SetTensorMetaSettings(bool inputEnabled, bool outputEnabled)
    {
        LOG_FUNC();
    
        if (m_inferType != DSL_INFER_TYPE_GIE)
        {
            LOG_ERROR("Unable to set tensor-meta settings for InferBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        if (IsLinked())
        {
            LOG_ERROR("Unable to set tensor-meta settings for InferBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_inputTensorMetaEnabled = inputEnabled;
        m_outputTensorMetaEnabled = outputEnabled;
        
        m_pInferEngine->SetAttribute("input-tensor-meta", m_inputTensorMetaEnabled);
        m_pInferEngine->SetAttribute("output-tensor-meta", m_outputTensorMetaEnabled);
        
        return true;
    }    

    bool InferBintr::SetRawOutputEnabled(bool enabled, const char* path)
    {
        LOG_FUNC();
        
        if (enabled)
        {
            struct stat info;

            if( stat(path, &info) != 0 )
            {
                LOG_ERROR("Unable to access path '" << path << "' for InferBintr '" 
                    << GetName() << "'");
                return false;
            }
            else if(info.st_mode & S_IFDIR)
            {
                LOG_INFO("Enabling raw layer-info output to path '" << path 
                    << "' for InferBintr '" << GetName() << "'");
                m_rawOutputPath.assign(path);
            }
            else
            {
                LOG_ERROR("Unable to access path '" << path 
                    << "' for InferBintr '" << GetName() << "'");
                return false;
            }
        }
        else
        {
            LOG_INFO("Disabling raw layer-info output to path '" << m_rawOutputPath 
                << "' for InferBintr '" << GetName() << "'");
            m_rawOutputPath.clear();
        }
        m_rawOutputEnabled = enabled;
        return true;
    }

    void InferBintr::HandleOnRawOutputGeneratedCB(GstBuffer* pBuffer, 
        NvDsInferNetworkInfo* pNetworkInfo, NvDsInferLayerInfo *pLayersInfo, 
        guint layersCount, guint batchSize)
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
                "_batch" + std::to_string(m_rawOutputFrameNumber) + 
                    "_bsize" + std::to_string(batchSize) + ".bin";
                
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
            streamOutputFile.write((char*)pLayerInfo->buffer, 
                typeSize * pLayerInfo->inferDims.numElements * batchSize);
            streamOutputFile.close();
        }
        m_rawOutputFrameNumber++;
    }

    bool InferBintr::AddModelUpdateListener(
        dsl_infer_engine_model_update_listener_cb listener, void* clientData)
    {
        LOG_FUNC();

        if (m_modelUpdateListeners.find(listener) != m_modelUpdateListeners.end())
        {   
            LOG_ERROR("Model Update listener is not unique");
            return false;
        }
        m_modelUpdateListeners[listener] = clientData;
        
        return true;
    }

    bool InferBintr::RemoveModelUpdateListener(
        dsl_infer_engine_model_update_listener_cb listener)
    {
        LOG_FUNC();
        
        if (m_modelUpdateListeners.find(listener) == m_modelUpdateListeners.end())
        {   
            LOG_ERROR("Model Update listener was not found");
            return false;
        }
        m_modelUpdateListeners.erase(listener);
        
        return true;
    }


    void InferBintr::HandleOnModelUpdatedCB(gchararray modelEngineFile)
    {
        LOG_FUNC();

        LOG_INFO("Model update complete for InferBintr '" 
            << GetName() << "'");
        LOG_INFO("New model = '" << modelEngineFile);

        if (m_modelUpdateListeners.size())
        {
            // Need the wstring version of the file path to send to the client.
            std::string cModelEngineFile(modelEngineFile);
            std::wstring wModeEngineFile(cModelEngineFile.begin(), 
                cModelEngineFile.end());

            // iterate through the map of listeners calling each
            for(auto const& imap: m_modelUpdateListeners)
            {
                try
                {
                    imap.first(GetWStrName().c_str(), 
                        wModeEngineFile.c_str(), imap.second);
                }
                catch(...)
                {
                    LOG_ERROR("Exception calling Client Model-Update-Lister");
                }
            }

        }

    }

    static void OnRawOutputGeneratedCB(GstBuffer* pBuffer, NvDsInferNetworkInfo* pNetworkInfo, 
        NvDsInferLayerInfo *pLayersInfo, guint layersCount, guint batchSize, gpointer pGie)
    {
        static_cast<InferBintr*>(pGie)->HandleOnRawOutputGeneratedCB(pBuffer, pNetworkInfo, 
            pLayersInfo, layersCount, batchSize);
    }


    static void OnModelUpdatedCB(GstElement* object, gint arg0, gchararray arg1,
        gpointer pInferBintr)
    {
        static_cast<InferBintr*>(pInferBintr)->HandleOnModelUpdatedCB(
            arg1);
    }

    // ***********************************************************************
    
    PrimaryInferBintr::PrimaryInferBintr(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, uint interval, uint inferType)
        : InferBintr(name, DSL_INFER_MODE_PRIMARY, inferConfigFile, 
            modelEngineFile, interval, inferType)
    {
        LOG_FUNC();
        
        AddChild(m_pInferEngine);

        // Float the queue element as a sink-ghost-pad for this Bintr.
        m_pQueue->AddGhostPadToParent("sink");

        // Float the infer-engine as a src-ghost-pad for this Bintr.
        m_pInferEngine->AddGhostPadToParent("src");
    }    
    
    PrimaryInferBintr::~PrimaryInferBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }

    bool PrimaryInferBintr::LinkAll()
    {
        LOG_FUNC();

        if (!m_batchSize)
        {
            LOG_ERROR("PrimaryInferBintr '" << GetName() 
                << "' can not be linked: batch size = 0");
            return false;
        }
        if (m_isLinked)
        {
            LOG_ERROR("PrimaryInferBintr '" << GetName() 
                << "' is already linked");
            return false;
        }
        if (!m_pQueue->LinkToSink(m_pInferEngine))
        {
            return false;
        }
        
        m_isLinked = true;
        
        return true;
    }
    
    void PrimaryInferBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("PrimaryInferBintr '" << GetName() << "' is not linked");
            return;
        }
        m_pQueue->UnlinkFromSink();

        m_isLinked = false;
    }

    bool PrimaryInferBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' PrimaryInferBintr to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddPrimaryInferBintr(shared_from_this());
    }

    bool PrimaryInferBintr::RemoveFromParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // remove 'this' PrimaryInfrBintr from the Parent Branch
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            RemovePrimaryInferBintr(shared_from_this());
    }

    // ***********************************************************************

    PrimaryAieBintr::PrimaryAieBintr(const char* name, 
        const char* inferConfigFile, const char* modelEngineFile,
        uint frameSize, uint hopSize, const char* transform)
        : PrimaryInferBintr(name, inferConfigFile, modelEngineFile, 
            0, DSL_INFER_TYPE_AIE)
        , m_hopSize(hopSize)
        , m_frameSize(frameSize) 
        , m_transform(transform)
    {
        LOG_FUNC();

        m_mediaType = DSL_MEDIA_TYPE_AUDIO_ONLY;

        // Set the frame and hop sizes
        m_pInferEngine->SetAttribute("audio-framesize", m_frameSize);
        m_pInferEngine->SetAttribute("audio-hopsize", m_hopSize);

        // Convert the transform string to a GstStructure
        GstStructure* pTransform = gst_structure_from_string(m_transform.c_str(), NULL);

        // Set the transform property and free the structure
        m_pInferEngine->SetAttribute("audio-transform", pTransform);
        gst_structure_free(pTransform);

        LOG_INFO("");
        LOG_INFO("Initial property values for PrimaryAieBintr '" << name << "'");
        LOG_INFO("  config-file-path     : " << m_inferConfigFile);
        LOG_INFO("  process-mode         : " << m_processMode);
        LOG_INFO("  unique-id            : " << m_uniqueId);
        LOG_INFO("  frame-size           : " << m_frameSize);
        LOG_INFO("  hop-size             : " << m_hopSize);
        LOG_INFO("  transfrom            : " << m_transform);
        LOG_INFO("  model-engine-file    : " << m_modelEngineFile);
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

    }

    PrimaryAieBintr::~PrimaryAieBintr()
    {
        LOG_FUNC();
    }

    uint PrimaryAieBintr::GetFrameSize()
    {
        LOG_FUNC();
        
        return m_frameSize;
    }

    bool PrimaryAieBintr::SetFrameSize(uint frameSize)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set frame-size for PrimaryAieBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        m_frameSize = frameSize;
        m_pInferEngine->SetAttribute("audio-framesize", m_frameSize);
        
        return true;
    }
    
    uint PrimaryAieBintr::GetHopSize()
    {
        LOG_FUNC();
        
        return m_hopSize;
    }

    bool PrimaryAieBintr::SetHopSize(uint hopSize)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set hop-size for PrimaryAieBintr '" << GetName() 
                << "' as it's currently linked");
            return false;
        }
        m_hopSize = hopSize;
        m_pInferEngine->SetAttribute("audio-hopsize", m_frameSize);
        
        return true;
    }
    
    const char* PrimaryAieBintr::GetTransform()
    {
        LOG_FUNC();
        
        return m_transform.c_str();
    }

    bool PrimaryAieBintr::SetTransform(const char* transform)
    {
        LOG_FUNC();
        
        m_transform.assign(transform);
        
        // Convert the transform string to a GstStructure
        GstStructure* pTransform = gst_structure_from_string(m_transform.c_str(), NULL);

        // Set the transform property and free the structure
        m_pInferEngine->SetAttribute("audio-transform", pTransform);
        gst_structure_free(pTransform);

        return true;
    }
    
    bool PrimaryAieBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();

        if (IsInUse())
        {
            LOG_ERROR("Unable to set GPU ID for PrimaryAieBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_gpuId = gpuId;

        m_pInferEngine->SetAttribute("gpu-id", m_gpuId);

        return true;
    }

    // ***********************************************************************

    PrimaryGieBintr::PrimaryGieBintr(const char* name, 
        const char* inferConfigFile, const char* modelEngineFile, uint interval)
        : PrimaryInferBintr(name, inferConfigFile, modelEngineFile, 
            interval, DSL_INFER_TYPE_GIE) 
    {
        LOG_FUNC();

        LOG_INFO("");
        LOG_INFO("Initial property values for InferBintr '" << name << "'");
        LOG_INFO("  Inference Type       : " << m_pInferEngine->GetFactoryName());
        LOG_INFO("  config-file-path     : " << m_inferConfigFile);
        LOG_INFO("  process-mode         : " << m_processMode);
        LOG_INFO("  unique-id            : " << m_uniqueId);
        LOG_INFO("  interval             : " << m_interval);
        LOG_INFO("  model-engine-file    : " << m_modelEngineFile);
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
    }

    PrimaryGieBintr::~PrimaryGieBintr()
    {
        LOG_FUNC();
    }

    bool PrimaryGieBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();

        if (IsInUse())
        {
            LOG_ERROR("Unable to set GPU ID for PrimaryGieBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_gpuId = gpuId;

        m_pInferEngine->SetAttribute("gpu-id", m_gpuId);

        return true;
    }


    // ***********************************************************************

    PrimaryTisBintr::PrimaryTisBintr(const char* name, 
        const char* inferConfigFile, uint interval)
        : PrimaryInferBintr(name, inferConfigFile, "", 
            interval, DSL_INFER_TYPE_TIS) 
    {
        LOG_FUNC();

        LOG_INFO("");
        LOG_INFO("Initial property values for PrimaryTisBintr '" << name << "'");
        LOG_INFO("  Inference Type       : " << m_pInferEngine->GetFactoryName());
        LOG_INFO("  config-file-path     : " << m_inferConfigFile);
        LOG_INFO("  process-mode         : " << m_processMode);
        LOG_INFO("  unique-id            : " << m_uniqueId);
        LOG_INFO("  interval             : " << m_interval);
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
    }
    
    PrimaryTisBintr::~PrimaryTisBintr()
    {
        LOG_FUNC();
    }

    // ***********************************************************************

    SecondaryInferBintr::SecondaryInferBintr(const char* name, 
        const char* inferConfigFile, const char* modelEngineFile, 
        const char* inferOn, uint interval, uint inferType)
        : InferBintr(name, DSL_INFER_MODE_SECONDARY, inferConfigFile, 
            modelEngineFile, interval, inferType)
        , m_inferOn(inferOn)
        , m_inferOnProcessMode(0)

    {
        LOG_FUNC();

        m_pTee = DSL_ELEMENT_NEW("tee", name);
        m_pFakeSinkQueue = DSL_ELEMENT_EXT_NEW("queue", name, "fakesink");
        
        m_pFakeSink = DSL_ELEMENT_NEW("fakesink", name);
        m_pFakeSink->SetAttribute("async", false);
        m_pFakeSink->SetAttribute("sync", false);
        m_pFakeSink->SetAttribute("enable-last-sample", false);
        
        LOG_INFO("  Infer on name     : " << inferOn);
        
        // Note: the Elementrs created/owned by this SecondaryInferBintr are added 
        // as children to the parent InferBintr, and not to this Bintr's GST BIN
        // In this way, all SecondaryInferBintrs Infer on the same buffer of data, 
        // regardlessof the depth of secondary Inference. Ghost Pads are not required 
        // for this bin

        // The Queue element is added as a child of this QBintr by default. Need to 
        // remove it so that it can be added as a child to the SecondaryInferBintrs as 
        // mentionded above.
        RemoveChild(m_pQueue);        
    }    
    
    SecondaryInferBintr::~SecondaryInferBintr()
    {
        LOG_FUNC();

        if (m_isLinked)
        {    
            UnlinkAll();
        }
    }

    bool SecondaryInferBintr::LinkAll()
    {
        LOG_FUNC();

        if (!m_batchSize)
        {
            LOG_ERROR("SecondaryInferBintr '" << GetName() 
                << "' can not be linked: batch size = 0");
            return false;
        }
        if (m_isLinked)
        {
            LOG_ERROR("SecondaryInferBintr '" << GetName() 
                << "' is already linked");
            return false;
        }
        if (!m_pQueue->LinkToSink(m_pInferEngine) or 
            !m_pInferEngine->LinkToSink(m_pTee) or
            !m_pFakeSinkQueue->LinkToSourceTee(m_pTee, "src_%u") or
            !m_pFakeSinkQueue->LinkToSink(m_pFakeSink))
        {
            LOG_ERROR("SecondaryInferBintr '" << GetName() 
                << "' failed to link");
            return false;
        }
        
        m_isLinked = true;
        
        return true;
    }
    
    void SecondaryInferBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("SecondaryInferBintr '" << GetName() 
                << "' is not linked");
            return;
        }
        m_pQueue->UnlinkFromSink();
        m_pInferEngine->UnlinkFromSink();
        m_pFakeSinkQueue->UnlinkFromSourceTee();
        m_pFakeSinkQueue->UnlinkFromSink();

        m_isLinked = false;
    }

    bool SecondaryInferBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // add 'this' GIE to the Parent Pipeline 
        return std::dynamic_pointer_cast<BranchBintr>(pParentBintr)->
            AddSecondaryInferBintr(shared_from_this());
    }
    
    uint SecondaryInferBintr::GetInferOnUniqueId()
    {
        LOG_FUNC();

        return m_inferOnUniqueId;
    }
    
    const char* SecondaryInferBintr::GetInferOnName()
    {
        LOG_FUNC();

        return m_inferOn.c_str();
    }
    
    uint SecondaryInferBintr::GetInferOnProcessMode()
    {
        LOG_FUNC();

        return m_inferOnProcessMode;
    }
    
    bool SecondaryInferBintr::SetInferOnAttributes()
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("Unable to update SecondaryInferBintr '" 
                << GetName() << "' as its in a linked state");
            return false;
            
        }
        if (Services::GetServices()->_inferAttributesGetByName(m_inferOn.c_str(),
            m_inferOnUniqueId, m_inferOnProcessMode) != DSL_RESULT_SUCCESS)
        {
            LOG_ERROR("SecondaryInferBintr '" << GetName() 
                << "' failed to Get unique Id for PrimaryInferBintr '" 
                << m_inferOn << "'");
            return false;
        }   
        
        LOG_INFO("Setting infer-on-id for SecondaryInferBintr '" 
            << GetName() << "' to " << m_inferOnUniqueId);
        
        m_pInferEngine->SetAttribute("infer-on-gie-id", m_inferOnUniqueId);
        
        return true;
    }

    bool SecondaryInferBintr::LinkToSource(DSL_NODETR_PTR pTee)
    {
        LOG_FUNC();

        LOG_INFO("Linking SecondaryInferBintr '" << GetName() 
            << "' to Tee '" << pTee->GetName() << "'");
        
        if (!m_pQueue->LinkToSource(pTee))
        {
            LOG_ERROR("Failed to link Tee '" << pTee->GetName() 
                << "' with SecondaryInferBintr '" << GetName() << "'");
            return false;
        }

        return true;
    }

    bool SecondaryInferBintr::UnlinkFromSource()
    {
        LOG_FUNC();
        
        // If we're not currently linked to the Tee
        if (!m_pQueue->IsLinkedToSource())
        {
            LOG_ERROR("SecondaryInferBintr '" << GetName() 
                << "' is not in a Linked state");
            return false;
        }

        if (!m_pQueue->UnlinkFromSource())
        {
            LOG_ERROR("SecondaryInferBintr '" << GetName() 
                << "' was not able to unlink from src Tee");
            return false;
        }
        LOG_INFO("Unlinking and releasing requested Source Pad for SecondaryInferBintr " 
            << GetName());
        
        return true;
    }

    SecondaryGieBintr::SecondaryGieBintr(const char* name,
        const char* inferConfigFile, const char* modelEngineFile, 
        const char* inferOn, uint interval)
        : SecondaryInferBintr(name, inferConfigFile, modelEngineFile, inferOn, 
            interval, DSL_INFER_TYPE_GIE)
    {
        LOG_FUNC();

        LOG_INFO("");
        LOG_INFO("Initial property values for SecondaryGieBintr '" << name << "'");
        LOG_INFO("  Inference Type       : " << m_pInferEngine->GetFactoryName());
        LOG_INFO("  config-file-path     : " << m_inferConfigFile);
        LOG_INFO("  unique-id            : " << m_uniqueId);
        LOG_INFO("  interval             : " << m_interval);
        LOG_INFO("  model-engine-file    : " << m_modelEngineFile);
        LOG_INFO("  infer-on             : " << m_inferOn);
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
    }

    SecondaryGieBintr::~SecondaryGieBintr()
    {
        LOG_FUNC();
    }

    // ***********************************************************************

    bool SecondaryGieBintr::SetGpuId(uint gpuId)
    {
        LOG_FUNC();
        
        if (IsInUse())
        {
            LOG_ERROR("Unable to set GPU ID for SecondaryGieBintr '" << GetName() 
                << "' as it's currently in use");
            return false;
        }

        m_gpuId = gpuId;
        m_pInferEngine->SetAttribute("gpu-id", m_gpuId);
        
        return true;
    }
    
    // ***********************************************************************

    SecondaryTisBintr::SecondaryTisBintr(const char* name,
        const char* inferConfigFile, const char* inferOn, uint interval)
        : SecondaryInferBintr(name, inferConfigFile, "", inferOn, 
            interval, DSL_INFER_TYPE_TIS)
    {
        LOG_FUNC();

        LOG_INFO("");
        LOG_INFO("Initial property values for SecondaryTisBintr '" << name << "'");
        LOG_INFO("  Inference Type       : " << m_pInferEngine->GetFactoryName());
        LOG_INFO("  config-file-path     : " << m_inferConfigFile);
        LOG_INFO("  unique-id            : " << m_uniqueId);
        LOG_INFO("  interval             : " << m_interval);
        LOG_INFO("  infer-on             : " << m_inferOn);
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
    }

    SecondaryTisBintr::~SecondaryTisBintr()
    {
        LOG_FUNC();
    }

}    

