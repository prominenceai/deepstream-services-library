

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

#ifndef _DSL_GIE_BINTR_H
#define _DSL_GIE_BINTR_H

#include "Dsl.h"
#include "DslBintr.h"
#include "DslElementr.h"

namespace DSL
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_PRIMARY_GIE_PTR std::shared_ptr<PrimaryGieBintr>
    #define DSL_PRIMARY_GIE_NEW(name, inferConfigFile, modelEngineFile, interval) \
        std::shared_ptr<PrimaryGieBintr>(new PrimaryGieBintr( \
        name, inferConfigFile, modelEngineFile, interval))

    #define DSL_SECONDARY_GIE_PTR std::shared_ptr<SecondaryGieBintr>
    #define DSL_SECONDARY_GIE_NEW(name, inferConfigFile, modelEngineFile, interval, inferOnGieName) \
        std::shared_ptr<SecondaryGieBintr>(new SecondaryGieBintr( \
        name, inferConfigFile, modelEngineFile, interval, inferOnGieName))

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
         * @param[in] inferConfigFile fully qualified pathspec for the infer config file to use
         * @param[in] modelEngineFile fully qualified pathspec for the model engine file to use
         * @param[in] interval
         */
        GieBintr(const char* name, const char* factoryname, uint processMode,
            const char* inferConfigFile, const char* modelEngineFile, uint interval);

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
         * @brief gets the name of the Model Engine File in use by this PrimaryGieBintr
         * @return fully qualified patspec used to create this Bintr
         */
        const char* GetModelEngineFile();
        
        /**
         * @brief sets the batch size for this Bintr
         * @param the new batchSize to use
         */
        void SetBatchSize(uint batchSize);
        
        /**
         * @brief gets the current batchSize in use by this PrimaryGieBintr
         * @return the current batchSize
         */
        uint GetBatchSize();

        /**
         * @brief sets the interval for this Bintr
         * @param the new interval to use
         */
        void SetInterval(uint interval);
        
        /**
         * @brief gets the current interval in use by this PrimaryGieBintr
         * @return the current interval setting
         */
        uint GetInterval();

        /**
         * @brief gets the current unique Id in use by this PrimaryGieBintr
         * @return the current unique Id
         */
        uint GetUniqueId();

    protected:

        /**
         * @brief pathspec to the infer config file used by this GIE
         */
        std::string m_inferConfigFile;
        
        /**
         * @brief pathspec to the model engine file used by this GIE
         */
        std::string m_modelEngineFile;
        
        /**
         * @brief 
         */
        uint m_batchSize;
        
        /**
         * @brief 
         */
        uint m_interval;
        
        /**
         * @brief 
         */
        uint m_uniqueId;

        /**
         @brief
         */
        uint m_processMode;

        /**
         * @brief Queue Elementr as Sink for this GieBintr
         */
        DSL_ELEMENT_PTR  m_pQueue;

        /**
         * @brief GST Infer Engine Elementr
         */
        DSL_ELEMENT_PTR  m_pInferEngine;

        DSL_ELEMENT_PTR  m_pFakeSink;

    };

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
         * @param[in] inferConfigFile fully qualified pathspec for the infer config file to use
         * @param[in] modelEnginFile fully qualified pathspec for the model engine file to use
         * @param[in] interval
         * @param[in] uniqueId
         */
        PrimaryGieBintr(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, uint interval);

        /**
         * @brief dtor for the PrimaryGieBintr
         */
        ~PrimaryGieBintr();

        /**
         * @brief Adds the PrimaryGieBintr to a Parent Pipeline Bintr
         * @param pParentBintr Parent Pipeline to add this Bintr to
         */
        bool AddToParent(DSL_NODETR_PTR pParentBintr);

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

    private:

        /**
         * @brief Video Converter Elementr for this PrimaryGieBintr
         */
        DSL_ELEMENT_PTR  m_pVidConv;
    

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
         * @param[in] inferConfigFile fully qualified pathspec for the infer config file to use
         * @param[in] modelEnginFile fully qualified pathspec for the model engine file to use
         * @param[in] interval
         * @param[in] inferOnGieName
         */
        SecondaryGieBintr(const char* name, const char* inferConfigFile,
            const char* modelEngineFile, uint interval, const char* inferOnGieName);

        /**
         * @brief dtor for the SecondaryGieBintr
         */
        ~SecondaryGieBintr();

        /**
         * @brief Adds the SecondaryGieBintr to a Parent Bintr
         * @param pParentBintr Parent to add this Bintr to
         */
        bool AddToParent(DSL_NODETR_PTR pParentBintr);

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
         * @param pTee that is source for this Secondary GIE
         * @return trueif this SGIE was able to link with source Tee, false otherwise
         */
        bool LinkToSource(DSL_NODETR_PTR pTee);

        /**
         * @brief Unlinks this SGIE's Queue Elementr from the previously linked-to source pTee
         * @param pTee that is source for this Secondary GIE
         * @return trueif this SGIE was able to unlink from the source Tee, false otherwise
         */
        bool UnlinkFromSource();
        
        /**
         * @brief returns the unique Id of the GIE this SGIE should infer on
         * @return unique Id for the Infer-on GIE
         */
        uint GetInferOnGieUniqueId();
        
        /**
         * @brief sets the Infer-on-GIE name for this Bintr
         * @param the new name of the GIE to infer on 
         */
        void SetInferOnGieName(const char* name);
        
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

    private:
        
        /**
         @brief Unique name of the Gie to infer on, primary or secondary
         */
        std::string m_inferOnGieName;
        
        /**
         @brief Unique Id of the Gie to infer on, primary or secondary
         */
        uint m_inferOnGieUniqueId;
        
    };
}

#endif // _DSL_GIE_BINTR_H