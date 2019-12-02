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

#ifndef _DSL_SGIES_BINTR_H
#define _DSL_SGIES_BINTR_H

#include "DslBintr.h"
#include "DslGieBintr.h"
    
   
namespace DSL 
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_PIPELINE_SGIES_PTR std::shared_ptr<PipelineSecondaryGiesBintr>
    #define DSL_PIPELINE_SGIES_NEW(name) \
        std::shared_ptr<PipelineSecondaryGiesBintr>(new PipelineSecondaryGiesBintr(name))

    /**
     * @class PipelineSecondaryGiesBintr
     * @brief Implements a container class for a collection of Secondary GIEs
     */
    class PipelineSecondaryGiesBintr : public Bintr
    {
    public: 
    
        PipelineSecondaryGiesBintr(const char* name);

        ~PipelineSecondaryGiesBintr();

        /**
         * @brief sets the current value of the Primary GIE Name for
         * which the first Secondary GIE will Infer-on
         * @param name name of the Primary GIE in use by the Parent Pipeline
         */
        void SetPrimaryGieName(const char* name);
        
        /**
         * @brief adds a child SecondaryGieBintr to this PipelineSecondaryGiesBintr
         * @param pChildSecondaryGie shared pointer to SecondaryGieBintr to add
         * @return true if the SecondaryGieBintr was added correctly, false otherwise
         */
        bool AddChild(DSL_SECONDARY_GIE_PTR pChildSecondaryGie);
        
        /**
         * @brief removes a child SecondaryGieBintr from this PipelineSecondaryGiesBintr
         * @param pChildSecondaryGie a shared pointer to SecondaryGieBintr to remove
         * @return true if the SecondaryGieBintr was removed correctly, false otherwise
         */
        bool RemoveChild(DSL_SECONDARY_GIE_PTR pChildSecondaryGie);

        /**
         * @brief overrides the base method and checks in m_pChildSecondaryGies only.
         */
        bool IsChild(DSL_SECONDARY_GIE_PTR pChildSecondaryGie);

        /**
         * @brief overrides the base Noder method to only return the number of 
         * child SecondaryGieBintrs and not the total number of children... 
         * i.e. exclude the nuber of child Elementrs from the count
         * @return the number of Child SecondaryGieBintrs held by this PipelineSecondaryGiesBintr
         */
        uint GetNumChildren()
        {
            LOG_FUNC();
            
            return m_pChildSecondaryGies.size();
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
         * @brief Sets the Batch Size for all child Secondary GIE's
         * @param batchSize value to set the Batch Size
         */
        void SetBatchSize(uint batchSize);
        
        /**
         * @brief 
         * @param pPad
         * @param pInfo
         * @return 
         */
        GstPadProbeReturn HandleSecondaryGiesSinkProbe(GstPad* pPad, GstPadProbeInfo* pInfo);
        
        /**
         * @brief 
         * @param pPad
         * @param pInfo
         * @return 
         */
        GstPadProbeReturn HandleSecondaryGiesSrcProbe(GstPad* pPad, GstPadProbeInfo* pInfo);

    private:
        /**
         * @brief adds a child Elementr to this PipelineSourcesBintr
         * @param pChildElement a shared pointer to the Elementr to add
         * @return a shared pointer to the Elementr if added correctly, nullptr otherwise
         */
        bool AddChild(DSL_NODETR_PTR pChildElement);
        
        /**
         * @brief removes a child Elementr from this PipelineSecondaryGiesBintr
         * @param pChildElement a shared pointer to the Elementr to remove
         */
        bool RemoveChild(DSL_NODETR_PTR pChildElement);
        

    public: // Members are public for the purpose of Test/Verification only

        DSL_ELEMENT_PTR m_pTee;
        DSL_ELEMENT_PTR m_pQueue;
    
        std::map<std::string, DSL_SECONDARY_GIE_PTR> m_pChildSecondaryGies;
        
        /**
         * @brief mutex to prevent sink (Tee) Pad Probe handler re-entry
         */
        GMutex m_sinkPadProbeMutex;
        
        /**
         * @brief mutex to prevent src (Queue) Pad Probe handler re-entry
         */
        GMutex m_srcPadProbeMutex;
        
        /**
         * @brief sink pad, Tee elementr, probe handle
         */
        uint m_sinkPadProbeId;

        /**
         * @brief src pad, Queue elementr, probe handle
         */
        uint m_srcPadProbeId;
        
        std::string m_primaryGieName;
        
        
        bool m_stop;

        bool m_flush;

        GCond m_padWaitLock;
        
    };

    /**
     * @brief Sink Pad Probe function 
     * @param pPad
     * @param pInfo
     * @param pGiesBintr
     * @return 
     */
    static GstPadProbeReturn SecondaryGiesSinkProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pGiesBintr);    
    
    /**
     * @brief Probe function to synchronize with the completion
     * of all Secondary GIEs working on the same buffer.
     * @param pPad
     * @param pInfo
     * @param pointer to a specific PipelineSecondaryGiesBintr
     * @return 
     */
    static GstPadProbeReturn SecondaryGiesSrcProbeCB(GstPad* pPad, 
        GstPadProbeInfo* pInfo, gpointer pGiesBintr);    

}

#endif // _DSL_SGIES_BINTR_H