

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
    #define DSL_GIE_PTR std::shared_ptr<GieBintr>

    #define DSL_PRIMARY_GIE_PTR std::shared_ptr<PrimaryGieBintr>
    #define DSL_PRIMARY_GIE_NEW(name, tritonEnabled, inferConfigFile, modelEngineFile, interval) \
        std::shared_ptr<PrimaryGieBintr>(new PrimaryGieBintr(name, \
        tritonEnabled, inferConfigFile, modelEngineFile, interval))

    #define DSL_SECONDARY_GIE_PTR std::shared_ptr<SecondaryGieBintr>
    #define DSL_SECONDARY_GIE_NEW(name, tritonEnabled, inferConfigFile, \
        modelEngineFile, inferOnGieName, interval) \
        std::shared_ptr<SecondaryGieBintr>(new SecondaryGieBintr(name, \
        tritonEnabled, inferConfigFile, modelEngineFile, inferOnGieName, interval))

    /**
     * @class GieBintr
     * @brief Implements a base class container for a GST Infer Engine (GIE)
     */
    class GieBintr : public Bintr
    {
    public: 
    
        /**
         * @brief ctor for the GST InferEngine
         * @param[in] name name to give the new Bintr
         * @param[in] tritonEnabled set to true to enable the Triton Inference Server
         * @param[in] inferConfigFile fully qualified pathspec for the infer config file to use
         * @param[in] modelEngineFile fully qualified pathspec for the model engine file to use
         * @param[in] interval
         */
        GieBintr(const char* name, bool tritonEnabled, uint processMode,
            const char* inferConfigFile, const char* modelEngineFile);

        /**
         * @brief dtor for the GieBintr
         */
        ~GieBintr();
        
        /**
         * @brief gets the name of the Infer Config File in use by this GieBintr
         * @return fully qualified patspec used to create this Bintr
         */
        const char* GetInferConfigFile();

        /**
         * @brief sets the name of the Infer Config File to use by this GieBintr
         * @return fully qualified patspec to the new Config File to use
         */
        bool SetInferConfigFile(const char* inferConfigFile);
        
        /**
         * @brief gets the name of the Model Engine File in use by this PrimaryGieBintr
         * @return fully qualified patspec used to create this Bintr
         */
        const char* GetModelEngineFile();

        /**
         * @brief sets the name of the Model Engine File to use by this GieBintr
         * @return fully qualified patspec to the new Model Engine File to use
         */
        bool SetModelEngineFile(const char* modelEngineFile);
        
        /**
         * @brief sets the batch size for this Bintr
         * @param the new batchSize to use
         */
        bool SetBatchSize(uint batchSize);
        
        /**
         * @brief sets the interval for this Bintr
         * @param the new interval to use
         */
        bool SetInterval(uint interval);
        
        /**
         * @brief gets the current interval in use by this PrimaryGieBintr
         * @return the current interval setting
         */
        uint GetInterval();

        /**
         * @brief gets the current unique Id in use by this PrimaryGieBintr
         * @return the current unique Id
         */
        int GetUniqueId();
        
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
        

    protected:

        /**
         * @brief helper function to generate a consistant Unique ID from string name
         * @param name string to generate the Unique ID from
         * @return numerical Unique ID
         */
        int CreateUniqueIdFromName(const char* name);
        
        /**
         * @brief pathspec to the infer config file used by this GIE
         */
        std::string m_inferConfigFile;
        
        /**
         * @brief pathspec to the model engine file used by this GIE
         */
        std::string m_modelEngineFile;
        
        /**
         * @brief current infer interval for the GieBintr
         */
        uint m_interval;
        
        /**
         * @brief Unique GIE ID derived from unique name
         */
        int m_uniqueId;
        
        /**
         * @brief True if the Triton Server pluggin is used/enabled
         */
        bool m_tritonEnabled;

        /**
         @brief Current process mode in use by the Primary
         */
        uint m_processMode;

        /**
         * @brief True if raw output is currently enabled.
         */
        bool m_rawOutputEnabled;
        
        /**
         * @brief pathspec to the raw output file directory used by this GIE
         */
        std::string m_rawOutputPath;
        
        /**
         * @brief maintains the current frame number between callbacks
         */
        ulong m_rawOutputFrameNumber;

        /**
         * @brief Queue Elementr as Sink for this GieBintr
         */
        DSL_ELEMENT_PTR  m_pQueue;

        /**
         * @brief GST Infer Engine Elementr
         */
        DSL_ELEMENT_PTR  m_pInferEngine;

        /**
         * @brief Fake sink used by GieBintr
         */
        DSL_ELEMENT_PTR  m_pFakeSink;

    };

    static void OnRawOutputGeneratedCB(GstBuffer* pBuffer, NvDsInferNetworkInfo* pNetworkInfo, 
        NvDsInferLayerInfo *pLayersInfo, guint layersCount, guint batchSize, gpointer pGie);

    /**
     * @class PrimaryGieBintr
     * @brief Implements a container for a Primary GST Infer Engine (GIE)
     */
    class PrimaryGieBintr : public GieBintr
    {
    public: 
    
        /**
         * @brief ctor for the PrimaryGieBintr
         * @param[in] name name to give the new Bintr
         * @param[in] tritonEnabled set to true to enable the Triton Inference Server
         * @param[in] inferConfigFile fully qualified pathspec for the infer config file to use
         * @param[in] modelEnginFile fully qualified pathspec for the model engine file to use
         * @param[in] interval GIE frame processing interval
         */
        PrimaryGieBintr(const char* name, bool tritonEnabled, const char* inferConfigFile,
            const char* modelEngineFile, uint interval);

        /**
         * @brief dtor for the PrimaryGieBintr
         */
        ~PrimaryGieBintr();

        /**
         * @brief Adds the PrimaryGieBintr to a Parent Pipeline Bintr
         * @param pParentBintr Parent Pipeline to add this Bintr to
         */
        bool AddToParent(DSL_BASE_PTR pParentBintr);

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
         * @brief Sets the GPU ID for all Elementrs
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);

    private:

        /**
         * @brief Video Converter Elementr for this PrimaryGieBintr
         */
        DSL_ELEMENT_PTR  m_pVidConv;
    
        /**
         * @brief Tee Elementr for this PrimaryGieBintr
         */
        DSL_ELEMENT_PTR  m_pTee;
    };

    /**
     * @class SecondaryGie
     * @brief Implements a container for a Secondary GST Infer Engine (GIE)
     */
    class SecondaryGieBintr : public GieBintr
    {
    public: 
    
        /**
         * @brief ctor for the SecondaryGieBintr
         * @param[in] name name to give the new Bintr
         * @param[in] tritonEnabled set to true to enable the Triton Inference Server
         * @param[in] inferConfigFile fully qualified pathspec for the infer config file to use
         * @param[in] modelEnginFile fully qualified pathspec for the model engine file to use
         * @param[in] frame interval to infer on
         */
        SecondaryGieBintr(const char* name, bool tritonEnable, const char* inferConfigFile,
            const char* modelEngineFile, const char* inferOnGieName, uint interval);

        /**
         * @brief dtor for the SecondaryGieBintr
         */
        ~SecondaryGieBintr();

        /**
         * @brief Adds the SecondaryGieBintr to a Parent Bintr
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
         * @brief Links this SGIE's Queue Elementr as sink back to the provided source pTee
         * @param[in] pTee that is source for this Secondary GIE
         * @return true if this SGIE was able to link with source Tee, false otherwise
         */
        bool LinkToSource(DSL_NODETR_PTR pTee);

        /**
         * @brief Unlinks this SGIE's Queue Elementr from the previously linked-to source pTee
         * @return true if this SGIE was able to unlink from the source Tee, false otherwise
         */
        bool UnlinkFromSource();
        
        /**
         * @brief returns the unique Id of the GIE this SGIE should infer on
         * @return unique Id for the Infer-on GIE
         */
        uint GetInferOnGieUniqueId();
        
        /**
         * @brief sets the Infer-on-GIE name for this Bintr
         * @param[in] the new name of the GIE to infer on 
         * @return true if this SGIE was able to set its Infer On GIE name, false otherwise
         */
        bool SetInferOnGieName(const char* name);
        
        /**
         * @brief gets the current Infer-on-GIE name in use by this SecondaryGieBintr
         * @return the current unique Id
         */
        const char* GetInferOnGieName();
        
        /**
         * @brief returns the Queue Elementr to the Parent Bintr of the SecondaryGieBintr
         * @return shared ponter to the Queue Elementr
         */
        DSL_ELEMENT_PTR GetQueueElementr()
        {
            LOG_FUNC();
            
            return m_pQueue;
        }

        /**
         * @brief returns the Infer Engine Elementr to the Parent Bintr of the SecondaryGieBintr
         * @return shared ponter to the Infer Engine Elementr
         */
        DSL_ELEMENT_PTR GetInferEngineElementr()
        {
            LOG_FUNC();
            
            return m_pInferEngine;
        }
        
        /**
         * @brief returns the Fake Sink Elementr to the Parent Bintr of the SecondaryGieBintr
         * @return shared ponter to the Fake Sink Elementr
         */
        DSL_ELEMENT_PTR GetFakeSinkElementr()
        {
            LOG_FUNC();
            
            return m_pFakeSink;
        }

        /**
         * @brief Sets the GPU ID for all Elementrs
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);
        
    private:
        
        /**
         @brief Unique name of the Gie to infer on, primary or secondary
         */
        std::string m_inferOnGieName;
        
        /**
         @brief Unique Id of the Gie to infer on, primary or secondary
         */
        int m_inferOnGieUniqueId;
        
    };
}

#endif // _DSL_GIE_BINTR_H