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

#ifndef _DSL_PROCESS_BINTR_H
#define _DSL_PROCESS_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBintr.h"
    
   
namespace DSL 
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_MULTI_SINKS_PTR std::shared_ptr<MultiSinksBintr>
    #define DSL_MULTI_SINKS_NEW(name) \
        std::shared_ptr<MultiSinksBintr>(new MultiSinksBintr(name))

    #define DSL_TEE_PTR std::shared_ptr<TeeBintr>
    #define DSL_TEE_NEW(name) \
        std::shared_ptr<TeeBintr>(new TeeBintr(name))

    #define DSL_STREAM_DEMUXER_PTR std::shared_ptr<StreamDemuxerBintr>
    #define DSL_STREAM_DEMUXER_NEW(name) \
        std::shared_ptr<StreamDemuxerBintr>(new StreamDemuxerBintr(name))

    /**
     * @class ProcessBintr
     * @brief 
     */
    class MultiComponentsBintr : public Bintr
    {
    public: 
    
        /**
         * @brief ctor for the MultiComponentsBintr
         * @param[in] name name to give the new Bintr
         */
        MultiComponentsBintr(const char* name, const char* teeType);

        /**
         * @brief dtor for the MultiComponentsBintr
         */
        ~MultiComponentsBintr();

        /**
         * @brief adds a child ComponentBintr to this MultiComponentsBintr
         * @param[in] pChildComponent shared pointer to ComponentBintr to add
         * @return true if the ComponentBintr was added correctly, false otherwise
         */
        bool AddChild(DSL_BINTR_PTR pChildComponent);
        
        /**
         * @brief removes a child ComponentBintr from this MultiComponentsBintr
         * @param[in] pChildComponent a shared pointer to ComponentBintr to remove
         * @return true if the ComponentBintr was removed correctly, false otherwise
         */
        bool RemoveChild(DSL_BINTR_PTR pChildComponent);

        /**
         * @brief overrides the base method and checks in m_pChildComponents only.
         */
        bool IsChild(DSL_BINTR_PTR pChildComponent);

        /**
         * @brief overrides the base Noder method to only return the number of 
         * child ComponentBintrs and not the total number of children... 
         * i.e. exclude the nuber of child Elementrs from the count
         * @return the number of Child ComponentBintrs held by this MultiComponentsBintr
         */
        uint GetNumChildren()
        {
            LOG_FUNC();
            
            return m_pChildComponents.size();
        }

        /** 
         * @brief links all child Component Bintrs and their elements
         */ 
        bool LinkAll();
        
        /**
         * @brief unlinks all child Component Bintrs and their Elementrs
         */
        void UnlinkAll();
        
        /**
         * @brief Links this MultiComponentsBintr back to a source Demuxer element
         * @param[in] pDemuxer to link back to
         * @return true on successful Link false other
         */
        bool LinkToSource(DSL_NODETR_PTR pDemuxer);
        
        /**
         * @brief Unlinks this MultiComponentsBintr from a source Demuxer element
         * @return true on successful Unlink false other
         */
        bool UnlinkFromSource();
        
        
    private:
    
        DSL_ELEMENT_PTR m_pQueue;
        DSL_ELEMENT_PTR m_pTee;
        
        /**
         * @brief Unique streamId of Parent SourceBintr if added to Source vs. Pipeline
         * The id is used when getting a request Pad for Src Demuxer
         */
        int m_streamId;
    
        std::map<std::string, DSL_BINTR_PTR> m_pChildComponents;

        /**
         * @brief A dynamic collection of requested Source Pads for this Bintr
         */
        std::map<std::string, GstPad*> m_pGstRequestedSourcePads;

        /**
         * @brief adds a child Elementr to this PipelineSourcesBintr
         * @param pChildElement a shared pointer to the Elementr to add
         * @return a shared pointer to the Elementr if added correctly, nullptr otherwise
         */
        bool AddChild(DSL_NODETR_PTR pChildElement);
        
        /**
         * @brief removes a child Elementr from this MultiComponentsBintr
         * @param pChildElement a shared pointer to the Elementr to remove
         */
        bool RemoveChild(DSL_NODETR_PTR pChildElement);

    };

    class MultiSinksBintr : public MultiComponentsBintr
    {
    public: 
    
        /**
         * @brief ctor for the MultiSinksBintr
         * @param[in] name name to give the new Bintr
         */
        MultiSinksBintr(const char* name);

    };

    class TeeBintr : public MultiComponentsBintr
    {
    public: 
    
        /**
         * @brief ctor for the MultiSinksBintr
         * @param[in] name name to give the new Bintr
         */
        TeeBintr(const char* name);

        /**
         * @brief Adds the MultiComponentBintr to a Parent Branch Bintr
         * @param[in] pParentBintr Parent Pipeline to add this Bintr to
         */
        bool AddToParent(DSL_NODETR_PTR pParentBintr);

    };

    class StreamDemuxerBintr : public MultiComponentsBintr
    {
    public: 
    
        /**
         * @brief ctor for the StreamDemuxerBintr
         * @param[in] name name to give the new Bintr
         */
        StreamDemuxerBintr(const char* name);

        /**
         * @brief Adds the MultiComponentBintr to a Parent Branch Bintr
         * @param[in] pParentBintr Parent Pipeline to add this Bintr to
         */
        bool AddToParent(DSL_NODETR_PTR pParentBintr);
    };

}

#endif // _DSL_PROCESS_BINTR_H