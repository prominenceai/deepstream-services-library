

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

#ifndef _DSL_GIE_BINTR_H
#define _DSL_GIE_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBintr.h"
#include "DslElementr.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_INFER_PTR std::shared_ptr<InferBintr>

    #define DSL_PRIMARY_INFER_PTR std::shared_ptr<PrimaryInferBintr>
    #define DSL_PRIMARY_INFER_NEW(name, inferConfigFile, \
        modelEngineFile, interval, inferType) \
        std::shared_ptr<PrimaryInferBintr>(new PrimaryInferBintr( \
            name, inferConfigFile, modelEngineFile, interval, inferType))

    #define DSL_PRIMARY_GIE_PTR std::shared_ptr<PrimaryGieBintr>
    #define DSL_PRIMARY_GIE_NEW(name, inferConfigFile, \
        modelEngineFile, interval) \
        std::shared_ptr<PrimaryGieBintr>(new PrimaryGieBintr( \
            name, inferConfigFile, modelEngineFile, interval))

    #define DSL_PRIMARY_TIS_PTR std::shared_ptr<PrimaryTisBintr>
    #define DSL_PRIMARY_TIS_NEW(name, inferConfigFile, interval) \
        std::shared_ptr<PrimaryTisBintr>(new PrimaryTisBintr( \
            name, inferConfigFile, interval))

    #define DSL_SECONDARY_INFER_PTR std::shared_ptr<SecondaryInferBintr>
    #define DSL_SECONDARY_INFER_NEW(name, inferConfigFile, \
        modelEngineFile, inferOn, interval, inferType) \
        std::shared_ptr<SecondaryInferBintr>(new SecondaryInferBintr( \
            name, inferConfigFile, modelEngineFile, inferOn, interval, inferType))

    #define DSL_SECONDARY_GIE_PTR std::shared_ptr<SecondaryGieBintr>
    #define DSL_SECONDARY_GIE_NEW(name, inferConfigFile, \
        modelEngineFile, inferOn, interval) \
        std::shared_ptr<SecondaryGieBintr>(new SecondaryGieBintr( \
            name, inferConfigFile, modelEngineFile, inferOn, interval))

    #define DSL_SECONDARY_TIS_PTR std::shared_ptr<SecondaryTisBintr>
    #define DSL_SECONDARY_TIS_NEW(name, inferConfigFile, inferOn, interval) \
        std::shared_ptr<SecondaryTisBintr>(new SecondaryTisBintr( \
            name, inferConfigFile, inferOn, interval))
            
    #define DSL_INFER_TYPE_GIE          0
    #define DSL_INFER_TYPE_TIS          1
    
    #define DSL_INFER_MODE_PRIMARY      1
    #define DSL_INFER_MODE_SECONDARY    2

    /**
     * @class InferBintr
     * @brief Implements a base class container for either a 
     * GST Inferece Engine (GIE) or Triton Inference Server (TIS)
     */
    class InferBintr : public Bintr
    {
    public: 
    
        /**
         * @brief ctor for the InferBintr
         * @param[in] name name to give the new Bintr
         * @param[in] inferConfigFile fully qualified pathspec for the infer 
         * config file to use
         * @param[in] modelEnginFile for GIS, fully qualified pathspec for the 
         * model engine file to use. Empty string = create model, 
         * TIS element does not have model property, use empty string 
         * @param[in] interval inference frame processing interval
         * @param[in] inferType either DSL_INFER_TYPE_GIE or DSL_INFER_TYPE_TIS
         */
        InferBintr(const char* name, uint processMode, const char* inferConfigFile, 
            const char* modelEngineFile, uint interval, uint inferType);

        /**
         * @brief dtor for the InferBintr
         */
        ~InferBintr();
        
        /**
         * @brief function to get the InferBintrs type as specified on creation
         * @return either DSL_INFER_TYPE_GIE or DSL_INFER_TYPE_TIS
         */
        uint GetInferType();
        
        /**
         * @brief gets the name of the Infer Config File in use by this InferBintr
         * @return fully qualified patspec used to create this Bintr
         */
        const char* GetInferConfigFile();

        /**
         * @brief sets the name of the Infer Config File to use by this InferBintr
         * @return fully qualified patspec to the new Config File to use
         */
        bool SetInferConfigFile(const char* inferConfigFile);
        
        /**
         * @brief gets the name of the Model Engine File in use by this InferBintr
         * @return fully qualified patspec used to create this Bintr
         */
        const char* GetModelEngineFile();

        /**
         * @brief sets the name of the Model Engine File to use by this InferBintr
         * @return fully qualified patspec to the new Model Engine File to use
         */
        bool SetModelEngineFile(const char* modelEngineFile);
        
        /**
         * @brief sets the batch size for this Bintr and sets m_batchSizeSetByClient
         * to true. Once set, calls to SetBatchSize will have no effect.
         * @param the new batchSize to use
         */
        bool SetBatchSizeByClient(uint batchSize);

        /**
         * @brief sets the batch size for this Bintr unless explicity set by
         * the client. 
         * @param the new batchSize to use
         */
        bool SetBatchSize(uint batchSize);
        
        /**
         * @brief sets the interval for this Bintr
         * @param the new interval to use
         */
        bool SetInterval(uint interval);
        
        /**
         * @brief gets the current interval in use by this InferBintr
         * @return the current interval setting
         */
        uint GetInterval();

        /**
         * @brief gets the current unique Id in use by this InferBintr
         * @return the current unique Id
         */
        int GetUniqueId();

        /**
         * @brief Gets the current input and output tensor-meta setting in use 
         * by this PrimaryGieBintr.
         * @param[out] inputEnabled if true preprocessing input tensors attached as 
         * metadata instead of preprocessing inside the plugin, false otherwise.
         * @param[out] outputEnabled if true tensor outputs will be attached as 
         * meta on GstBuffer.
         */
        void GetTensorMetaSettings(bool* inputEnabled, bool* outputEnabled);

        /**
         * @brief Gets the current input and output tensor-meta setting in use 
         * by this PrimaryGieBintr.
         * @param[in] input_enabled set to true preprocess input tensors attached as 
         * metadata instead of preprocessing inside the plugin, false otherwise.
         * @param[in] output_enabled set to true to attach tensor outputs as 
         * meta on GstBuffer.
         * @return true if successfully set, false otherwise.
         */
        bool SetTensorMetaSettings(bool inputEnabled, bool outputEnabled);

        /**
         * @brief Enables/disables raw NvDsInferLayerInfo to .bin file.
         * @param enabled true if info should be written to file, false to disable
         * @param path relative or absolute dir path specification
         * @return true if success, false otherwise.
         */
        bool SetRawOutputEnabled(bool enabled, const char* path);
        
        /**
         * @brief Writes raw layer info to bin file
         * @param buffer - currently not used
         * @param networkInfo - not used
         * @param layersInfo layer information to write out
         * @param layersCount number of layer info structures
         * @param batchSize batch-size set to number of sources
         */
        void HandleOnRawOutputGeneratedCB(GstBuffer* pBuffer, NvDsInferNetworkInfo* pNetworkInfo, 
        NvDsInferLayerInfo *pLayersInfo, guint layersCount, guint batchSize);
        
        /**
         * @brief static list of unique Infer plugin IDs to be used/recycled by all
         * InferBintrs ctor/dtor
         */
        static std::list<uint> s_uniqueIds;

    protected:
    
        /**
         * @breif either DSL_INFER_TYPE_GIE or DSL_INFER_TYPE_TIS
         */
        uint m_inferType;
        
        /**
         * @brief pathspec to the infer config file used by this InferBintr
         */
        std::string m_inferConfigFile;
        
        /**
         * @brief pathspec to the model engine file used by this InferBintr
         */
        std::string m_modelEngineFile;
        
        /**
         * @brief true if Client explicity set by client, false by default.
         */
        bool m_batchSizeSetByClient;
        
        /**
         * @brief current infer interval for the InferBintr
         */
        uint m_interval;
        
        /**
         * @brief Unique ID derived from unique name
         */
        int m_uniqueId;

        /**
         @brief Current process mode in use by the Primary
         */
        uint m_processMode;

        /**
         * true if raw output is currently enabled.
         */
        bool m_rawOutputEnabled;
        
        /**
         * @brief pathspec to the raw output file directory used by this InferBintr
         */
        std::string m_rawOutputPath;
        
        /**
         * @brief maintains the current frame number between callbacks
         */
        ulong m_rawOutputFrameNumber;
        
        /**
         * @brief current input-temsor-meta enabled setting for this InferBintr.
         * NOTE: only used by the GIE Binters at this time
         */
        bool m_inputTensorMetaEnabled;
        
        /**
         * @brief current output-temsor-meta enabled setting for this InferBintr.
         * NOTE: only used by the GIE Binters at this time
         */
        bool m_outputTensorMetaEnabled;

        /**
         * @brief Queue Elementr as Sink for this InferBintr
         */
        DSL_ELEMENT_PTR  m_pQueue;

        /**
         * @brief Infer Elementr, either GIE or TIS
         */
        DSL_ELEMENT_PTR  m_pInferEngine;

    };

    static void OnRawOutputGeneratedCB(GstBuffer* pBuffer, NvDsInferNetworkInfo* pNetworkInfo, 
        NvDsInferLayerInfo *pLayersInfo, guint layersCount, guint batchSize, gpointer pGie);

    /**
     * @class PrimaryInferBintr
     * @brief Implements a container for a Primary GIE or TIS
     */
    class PrimaryInferBintr : public InferBintr
    {
    public: 
    
        /**
         * @brief ctor for the PrimaryInferBintr
         * @param[in] name name to give the new Bintr
         * @param[in] inferConfigFile fully qualified pathspec for the infer 
         * config file to use
         * @param[in] modelEnginFile for GIS, fully qualified pathspec for the 
         * model engine file to use. Empty string = create model, 
         * TIS element does not have model property, use empty string 
         * @param[in] interval inference frame processing interval
         */
        PrimaryInferBintr(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, uint interval, uint inferType);

        /**
         * @brief dtor for the PrimaryInferBintr
         */
        ~PrimaryInferBintr();

        /**
         * @brief Adds the PrimaryInferBintr to a Parent Branch Bintr
         * @param pParentBintr Parent Branch to add this Bintr to
         */
        bool AddToParent(DSL_BASE_PTR pParentBintr);

        /**
         * @brief Removes this PrimaryInferBintr from its Parent Branch Bintr
         * @param[in] pParentBintr parent Pipeline to remove from
         * @return true on successful add, false otherwise
         */
        bool RemoveFromParent(DSL_BASE_PTR pParentBintr);

        /**
         * @brief Links all Child Elementrs owned by this Bintr
         * @return true if all links were succesful, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elementrs owned by this Bintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();

        /**
         * @brief Sets the NVIDIA buffer memory type.
         * @brief nvbufMemType new memory type to use, on of the 
         * DSL_NVBUF_MEM_TYPE constant values.
         * @return true if successfully set, false otherwise.
         */
        bool SetNvbufMemType(uint nvbufMemType);

    protected:

        /**
         * @brief Video Converter Elementr for this PrimaryInferBintr
         */
        DSL_ELEMENT_PTR  m_pVidConv;
    
        /**
         * @brief Tee Elementr for this PrimaryInferBintr
         */
        DSL_ELEMENT_PTR  m_pTee;
    };

    // ***********************************************************************

    /**
     * @class PrimaryGieBintr
     * @brief Implements a container for a Primary GIE
     */
    class PrimaryGieBintr : public PrimaryInferBintr
    {
    public: 
    
        /**
         * @brief ctor for the PrimaryGieBintr
         * @param[in] name name to give the new Bintr
         * @param[in] inferConfigFile fully qualified pathspec for the infer 
         * config file to use
         * @param[in] modelEnginFile fully qualified pathspec for the 
         * model engine file to use. Empty string = create model
         * @param[in] interval inference frame processing interval
         */
        PrimaryGieBintr(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, uint interval);

        /**
         * @brief dtor for the PrimaryGieBintr
         */
        ~PrimaryGieBintr();

        /**
         * @brief Sets the GPU ID for all Elementrs owned by this PrimaryGieBintr
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);
        
    };

    // ***********************************************************************

    /**
     * @class PrimaryTisBintr
     * @brief Implements a container for a Primary Triton Inference Server (PTIS)
     */
    class PrimaryTisBintr : public PrimaryInferBintr
    {
    public: 
    
        /**
         * @brief ctor for the PrimaryTisBintr
         * @param[in] name name to give the new Bintr
         * @param[in] inferConfigFile fully qualified pathspec for the infer 
         * config file to use
         * @param[in] interval inference frame processing interval
         */
        PrimaryTisBintr(const char* name, const char* inferConfigFile,
            uint interval);

        /**
         * @brief dtor for the PrimaryTisBintr
         */
        ~PrimaryTisBintr();
    };

    // ***********************************************************************

    /**
     * @class SecondaryInferBintr
     * @brief Implements a container for a Secondary GIE or TIS
     */
    class SecondaryInferBintr : public InferBintr
    {
    public: 
    
        /**
         * @brief ctor for the SecondaryInferBintr
         * @param[in] name name to give the new Bintr
         * @param[in] inferConfigFile fully qualified pathspec for the infer 
         * config file to use
         * @param[in] modelEnginFile for GIS, fully qualified pathspec for the 
         * model engine file to use. Empty string = create model, 
         * TIS element does not have model property, use empty string 
         * @param[in] inferOn name of the Primary or Secondary InferBintr's
         * output to perform inference on.
         * @param[in] interval frame interval to infer on
         */
        SecondaryInferBintr(const char* name, const char* inferConfigFile, 
            const char* modelEngineFile, const char* inferOn, 
            uint interval, uint inferType);

        /**
         * @brief dtor for the SecondaryInferBintr
         */
        ~SecondaryInferBintr();

        /**
         * @brief Adds the SecondaryInferBintr to a Parent Bintr
         * @param[in] pParentBintr Parent to add this Bintr to
         */
        bool AddToParent(DSL_BASE_PTR pParentBintr);

        /**
         * @brief Links all Child Elementrs owned by this Bintr
         * @return true if all links were succesful, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elemntrs owned by this Bintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();
        
        /**
         * @brief Links this Bintr's Queue Elementr as sink back to the provided source pTee
         * @param[in] pTee that is source for this Secondary GIE
         * @return true if this Bintr was able to link with source Tee, false otherwise
         */
        bool LinkToSource(DSL_NODETR_PTR pTee);

        /**
         * @brief Unlinks this Bintr's Queue Elementr from the previously linked-to source pTee
         * @return true if this Bintr was able to unlink from the source Tee, false otherwise
         */
        bool UnlinkFromSource();
        
        /**
         * @brief returns the unique Id of the InferBintr this SecondarInferBintr 
         * should infer on
         * @return unique Id for the Infer-on Bintr
         */
        uint GetInferOnUniqueId();
        
        /**
         * @brief sets the Infer-on name for this Bintr
         * @param[in] the new name of the Infer Bintr to infer on 
         * @return true if this Bintr was able to set its Infer-on name, false otherwise
         */
        bool SetInferOnName(const char* name);
        
        /**
         * @brief gets the current Infer-on name in use by this SecondaryInferBintr
         * @return the current unique Id
         */
        const char* GetInferOnName();
        
        /**
         * @brief returns the Queue Elementr to the Parent Bintr of the SecondaryInferBintr
         * @return shared ponter to the Queue Elementr
         */
        DSL_ELEMENT_PTR GetQueueElementr()
        {
            LOG_FUNC();
            
            return m_pQueue;
        }

        /**
         * @brief returns the Infer Engine Elementr to the Parent Bintr of the SecondaryInferBintr
         * @return shared ponter to the Infer Engine Elementr
         */
        DSL_ELEMENT_PTR GetInferEngineElementr()
        {
            LOG_FUNC();
            
            return m_pInferEngine;
        }

        /**
         * @brief returns the Tee Elementr to the Parent Bintr of this SecondaryInferBintr
         * @return shared ponter to the Fake Sink Elementr
         */
        DSL_ELEMENT_PTR GetTeeElementr()
        {
            LOG_FUNC();
            
            return m_pTee;
        }

        /**
         * @brief returns the Queue Elementr to the Parent Bintr of the SecondaryInferBintr
         * @return shared ponter to the Queue Elementr
         */
        DSL_ELEMENT_PTR GetFakeSinkQueueElementr()
        {
            LOG_FUNC();
            
            return m_pFakeSinkQueue;
        }

        /**
         * @brief returns the Fake Sink Elementr to the Parent Bintr of this SecondaryInferBintr
         * @return shared ponter to the Fake Sink Elementr
         */
        DSL_ELEMENT_PTR GetFakeSinkElementr()
        {
            LOG_FUNC();
            
            return m_pFakeSink;
        }

    private:
        
        /**
         @brief Unique name of the InferBintr to infer on, primary or secondary
         */
        std::string m_inferOn;
        
        /**
         @brief Unique Id of the InferBintr to infer on, primary or secondary
         */
        int m_inferOnUniqueId;
        
        /**
         * @brief Tee Elementr for this SecondaryInferBintr
         */
        DSL_ELEMENT_PTR  m_pTee;

        /**
         * @brief Queue Elementr between Tee and FakeSink for this SecondaryInferBintr
         */
        DSL_ELEMENT_PTR  m_pFakeSinkQueue;

        /**
         * @brief Fake sink used by InferBintr
         */
        DSL_ELEMENT_PTR  m_pFakeSink;

    };

    // ***********************************************************************

    /**
     * @class SecondaryGieBintr
     * @brief Implements a container for a Secondary GIE
     */
    class SecondaryGieBintr : public SecondaryInferBintr
    {
    public: 
    
        /**
         * @brief ctor for the SecondaryGieBintr
         * @param[in] name name to give the new Bintr
         * @param[in] inferConfigFile fully qualified pathspec for the infer 
         * config file to use
         * @param[in] modelEnginFile fully qualified pathspec for the 
         * model engine file to use. Empty string = create model, 
         * @param[in] inferOn name of the Primary or Secondary InferBintr's
         * output to perform inference on.
         * @param[in] interval frame interval to infer on
         */
        SecondaryGieBintr(const char* name, const char* inferConfigFile, 
            const char* modelEngineFile, const char* inferOn, uint interval);

        /**
         * @brief dtor for the SecondaryGieBintr
         */
        ~SecondaryGieBintr();

        /**
         * @brief Sets the GPU ID for all Elementrs owned by this SecondaryGieBintr
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);
    };    

    // ***********************************************************************
    
    /**
     * @class SecondaryTisBintr
     * @brief Implements a container for a Secondary TIS
     */
    class SecondaryTisBintr : public SecondaryInferBintr
    {
    public: 
    
        /**
         * @brief ctor for the SecondaryTisBintr
         * @param[in] name name to give the new Bintr
         * @param[in] inferConfigFile fully qualified pathspec for the infer 
         * config file to use
         * @param[in] inferOn name of the Primary or Secondary InferBintr's
         * output to perform inference on.
         * @param[in] interval frame interval to infer on
         */
        SecondaryTisBintr(const char* name, const char* inferConfigFile, 
            const char* inferOn, uint interval);

        /**
         * @brief dtor for the SecondaryTisBintr
         */
        ~SecondaryTisBintr();
    };
}

#endif // _DSL_GIE_BINTR_H