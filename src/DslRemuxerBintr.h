/*
The MIT License

Copyright (c) 2021, Prominence AI, Inc.

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

#ifndef _DSL_REMUXER_BINTR_H
#define _DSL_REMUXER_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslElementr.h"
#include "DslMultiSourcesBintr.h"
#include "DslMultiBranchesBintr.h"
#include "DslBintr.h"

namespace DSL
{
    
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_REMUXER_PTR std::shared_ptr<RemuxerBintr>
    #define DSL_REMUXER_NEW(name, maxStreamIds) \
        std::shared_ptr<RemuxerBintr>(new RemuxerBintr(name, \
            maxStreamIds))

    /**
     * @class RemuxerBintr
     * @brief Implements a Remuxer (demuxer-streammuxer) bin container
     */
    class RemuxerBintr : public Bintr
    {
    public: 
    
        /**
         * @brief ctor for the RemuxerBintr class
         * @param[in] name name to give the new RemuxerBintr
         * @param[in] maxStreamIds maximum number of streams that can be connect.
         */
        RemuxerBintr(const char* name, uint maxStreamIds);

        /**
         * @brief dtor for the RemuxerBintr class
         */
        ~RemuxerBintr();

        /**
         * @brief Adds this RemuxerBintr to a Parent Pipeline or Branch Bintr.
         * @param[in] pParentBintr parent Pipeline to add to
         * @return true on successful add, false otherwise
         */
        bool AddToParent(DSL_BASE_PTR pParentBintr);
        
        /**
         * @brief Removes this RemuxerBintr from a Parent Pipeline or Branch Bintr.
         * @param[in] pParentBintr parent Pipeline or Branch to remove from.
         * @return true on successful add, false otherwise
         */
        bool RemoveFromParent(DSL_BASE_PTR pParentBintr);

        /**
         * @brief Adds a child ComponentBintr to this RemuxerBintr. Each child 
         * (branch) is assigned a new Streammuxer. Each Streammuxer is connected
         * to multiple streams produced by, and tee'd off of, the Bintr's Demuxer.
         * @param[in] pChildComponent shared pointer to ComponentBintr (branch) to add.
         * @param[in] streamIds array of streamIds - identifing which streams to connect
         * to.
         * @return true if the ComponentBintr was added correctly, false otherwise
         */
        bool AddChild(DSL_BINTR_PTR pChildComponent, 
            uint* streamIds, uint numStreamIds);

        /**
         * @brief removes a child ComponentBintr from this RemuxerBintr
         * @param[in] pChildComponent a shared pointer to ComponentBintr to remove.
         * @return true if the ComponentBintr was removed correctly, false otherwise
         */
        bool RemoveChild(DSL_BINTR_PTR pChildComponent);
        
        /**
         * @brief overrides the base method and checks in m_pChildBranches only.
         */
        bool IsChild(DSL_BINTR_PTR pChildComponent);

        /**
         * @brief overrides the base Noder method to only return the number of 
         * child ComponentBintrs and not the total number of children... 
         * i.e. exclude the nuber of child Elementrs from the count
         * @return the number of Child ComponentBintrs (branches) held by this 
         * MultiBranchesBintr
         */
        uint GetNumChildren()
        {
            LOG_FUNC();
            
            return m_pChildBranches.size();
        }

        /**
         * @brief Links all child elements of this RemuxerBintr
         * @return true if all elements were succesfully linked, false otherwise.
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all child elements of the RemuxerBintr
         */
        void UnlinkAll();

        /**
         * @brief Sets the GPU ID for all Elementrs
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);

    private:
    
    
        /**
         * @brief Number of stream-ids either currently or to be connected.
         */
  //      uint m_numStreamIds;
        
        /**
         * @brief Maximum number of stream-ids that can be connected.
         */
        uint m_maxStreamIds;
        
        /**
         * @brief Vector of stream-ids enabled for remuxing.
         */
//        std::vector<uint> m_streamIds;

        /**
         * @brief Streamdemuxer child Bintr for the RemuxerBintr
         */
        DSL_DEMUXER_PTR m_pDemuxer;
        
        /**
         * @brief Vector of Splitter Tees, one per maxStreamIds, for the RemuxerBintr.
         */
        std::vector<DSL_SPLITTER_PTR> m_splitters;

        /**
         * @brief container of all child Sinks/Branches mapped by their unique names
         */
        std::map<std::string, DSL_BINTR_PTR> m_pChildBranches;
        
        /**
         * @brief Map of child Streammuxers, one per child Branch, 
         * for the RemuxerBintr.
         */
        std::map<std::string, DSL_MULTI_SOURCES_PTR> m_muxers;

        /**
         * @brief Map of SourceQueue Vectors for the RemuxerBintr. Entries are
         * mapped by unique Streammuxer Bintr name.
         */
        std::map<std::string, std::vector<DSL_QUEUE_SOURCE_PTR>> m_pSourceQueues;
    
    };
    
}

#endif // _DSL_REMUXER_BINTR_H