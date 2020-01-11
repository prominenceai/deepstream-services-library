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

#include "DslBintr.h"
#include "DslSinkBintr.h"
    
   
namespace DSL 
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_MULTI_SINKS_PTR std::shared_ptr<MultiSinksBintr>
    #define DSL_MULTI_SINKS_NEW(name) \
        std::shared_ptr<MultiSinksBintr>(new MultiSinksBintr(name))

    /**
     * @class ProcessBintr
     * @brief 
     */
    class MultiSinksBintr : public Bintr
    {
    public: 
    
        MultiSinksBintr(const char* name);

        ~MultiSinksBintr();
        
        /**
         * @brief adds a child SinkBintr to this MultiSinksBintr
         * @param pChildSink shared pointer to SinkBintr to add
         * @return true if the SinkBintr was added correctly, false otherwise
         */
        bool AddChild(DSL_SINK_PTR pChildSink);
        
        /**
         * @brief removes a child SinkBintr from this MultiSinksBintr
         * @param pChildSink a shared pointer to SinkBintr to remove
         * @return true if the SinkBintr was removed correctly, false otherwise
         */
        bool RemoveChild(DSL_SINK_PTR pChildSink);

        /**
         * @brief overrides the base method and checks in m_pChildSinks only.
         */
        bool IsChild(DSL_SINK_PTR pChildSink);

        /**
         * @brief overrides the base Noder method to only return the number of 
         * child SinkBintrs and not the total number of children... 
         * i.e. exclude the nuber of child Elementrs from the count
         * @return the number of Child SinkBintrs held by this MultiSinksBintr
         */
        uint GetNumChildren()
        {
            LOG_FUNC();
            
            return m_pChildSinks.size();
        }

        /** 
         * @brief links all child Sink Bintrs and their elements
         */ 
        bool LinkAll();
        
        /**
         * @brief unlinks all child Sink Bintrs and their Elementrs
         */
        void UnlinkAll();
        
        
    private:
        DSL_ELEMENT_PTR m_pQueue;
        DSL_ELEMENT_PTR m_pTee;
    
        std::map<std::string, DSL_SINK_PTR> m_pChildSinks;

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
         * @brief removes a child Elementr from this MultiSinksBintr
         * @param pChildElement a shared pointer to the Elementr to remove
         */
        bool RemoveChild(DSL_NODETR_PTR pChildElement);

    };
}

#endif // _DSL_PROCESS_BINTR_H