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

#ifndef _DSL_SINFERS_BINTR_H
#define _DSL_SINFERS_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBintr.h"
#include "DslInferBintr.h"
    
   
namespace DSL 
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_PIPELINE_SINFERS_PTR std::shared_ptr<PipelineSInfersBintr>
    #define DSL_PIPELINE_SINFERS_NEW(name) \
        std::shared_ptr<PipelineSInfersBintr>(new PipelineSInfersBintr(name))

    /**
     * @class PipelineSInfersBintr
     * @brief Implements a container class for a collection of 
     * Secondary GIEs or TIEs - types cannot be mixed
     */
    class PipelineSInfersBintr : public Bintr
    {
    public: 
    
        PipelineSInfersBintr(const char* name);

        ~PipelineSInfersBintr();

        /**
         * @brief adds a child SecondaryInferBintr to this PipelineSInfersBintr
         * @param pChildSecondaryInfer shared pointer to SecondaryInferBintr to add
         * @return true if the SecondaryInferBintr was added correctly, false otherwise
         */
        bool AddChild(DSL_SECONDARY_INFER_PTR pChildSecondaryInfer);
        
        /**
         * @brief removes a child SecondaryInferBintr from this PipelineSInfersBintr
         * @param pChildSecondaryInfr a shared pointer to SecondaryInferBintr to remove
         * @return true if the SecondaryInferBintr was removed correctly, false otherwise
         */
        bool RemoveChild(DSL_SECONDARY_INFER_PTR pChildSecondaryInfer);

        /**
         * @brief overrides the base method and checks in m_pChildSecondaryGies only.
         */
        bool IsChild(DSL_SECONDARY_INFER_PTR pChildSecondaryInfer);

        /**
         * @brief overrides the base Noder method to only return the number of 
         * child SecondaryInferBintrs and not the total number of children... 
         * i.e. exclude the nuber of child Elementrs from the count
         * @return the number of Child SecondaryInferBintrs held by this PipelineSInfersBintr
         */
        uint GetNumChildren()
        {
            LOG_FUNC();
            
            return m_pChildSInfers.size();
        }

        /** 
         * @brief links all child Sink Bintrs and their elements
         */ 
        bool LinkAll();
        
        /**
         * @brief unlinks all child Sink Bintrs and their Elementrs
         */
        void UnlinkAll();

        /**
         * @brief sets the current value of the Primary GIE Name for
         * which the first Secondary GIE will Infer-on
         * @param id unique id of the Primary GIE in use by the Parent Pipeline
         */
        void SetInferOnId(int id);
        
        /**
         * @brief Gets the Interval for all child Secondary GIE's
         * @return the current Interval setting 
         */
        uint GetInterval();
        
        /**
         * @brief Sets the Batch Size for all child Secondary GIE's
         * @param batchSize value to set the Batch Size
         */
        void SetInterval(uint interval);
        
        /**
         * @brief 
         * @param pPad
         * @param pInfo
         * @return 
         */
        GstPadProbeReturn HandleSInfersSinkProbe(GstPad* pPad, GstPadProbeInfo* pInfo);
        
        /**
         * @brief 
         * @param pPad
         * @param pInfo
         * @return 
         */
        GstPadProbeReturn HandleSInfersSrcProbe(GstPad* pPad, GstPadProbeInfo* pInfo);

    private:
        /**
         * @brief adds a child Elementr to this PipelineSInfersBintr
         * @param pChildElement a shared pointer to the Elementr to add
         * @return a shared pointer to the Elementr if added correctly, nullptr otherwise
         */
        bool AddChild(DSL_BASE_PTR pChildElement);
        
        /**
         * @brief removes a child Elementr from this PipelineSInfersBintr
         * @param pChildElement a shared pointer to the Elementr to remove
         */
        bool RemoveChild(DSL_BASE_PTR pChildElement);

        /**
         * @brief Tee's the output from the Primary GIE as input for all 
         * 2nd-level SecondaryInferBintrs. 2nd-level SecondaryInferBintrs create their own Tees for all
         * for any third level SecondaryInferBintrs, and so on.
         */
        DSL_ELEMENT_PTR m_pTee;
        
        /**
         * @brief All SecondaryInferBintrs, at all levels, infer on the same buffer of data
         * which get's queued by this Elementr. A Pad Probe is used to block the
         * output of the Queue until all inference by all SecondaryInferBintrs is complete.
         */
        DSL_ELEMENT_PTR m_pQueue;
    
        /**
         * @brief map of all child SecondaryInferBintrs keyed by their unique component name
         */
        std::map<std::string, DSL_SECONDARY_INFER_PTR> m_pChildSInfers;
        
        /**
         * @brief mutex for sink (Tee) Pad Probe handler
         */
        DslMutex m_sinkPadProbeMutex;
        
        /**
         * @brief mutex fo the src (Queue) Pad Probe handler
         */
        DslMutex m_srcPadProbeMutex;
        
        /**
         * @brief sink pad, Tee elementr, probe handle
         */
        uint m_sinkPadProbeId;

        /**
         * @brief src pad, Queue elementr, probe handle
         */
        uint m_srcPadProbeId;
        
        uint m_interval;

        bool m_stop;

        bool m_flush;

        GCond m_padWaitLock;
    };

    /**
     * @brief Sink Pad Probe function 
     * @param pPad
     * @param pInfo
     * @param pSInfersBintr
     * @return 
     */
    static GstPadProbeReturn SInfersSinkProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pSInfersBintr);    
    
    /**
     * @brief Probe function to synchronize with the completion
     * of all Secondary Infers working on the same buffer.
     * @param pPad
     * @param pInfo
     * @param pointer to a specific PipelineSInfersBintr
     * @return 
     */
    static GstPadProbeReturn SInfersSrcProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pSInfersBintr);    

}

#endif // _DSL_SINFERS_BINTR_H