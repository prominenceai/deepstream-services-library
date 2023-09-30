/*
The MIT License

Copyright (c) 2019-2023, Prominence AI, Inc.

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
#include "DslPadProbeHandler.h"
   
namespace DSL 
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_TEE_PTR std::shared_ptr<TeeBintr>

    #define DSL_MULTI_BRANCHES_PTR std::shared_ptr<MultiBranchesBintr>

    #define DSL_MULTI_SINKS_PTR std::shared_ptr<MultiSinksBintr>
    #define DSL_MULTI_SINKS_NEW(name) \
        std::shared_ptr<MultiSinksBintr>(new MultiSinksBintr(name))

    #define DSL_DEMUXER_PTR std::shared_ptr<DemuxerBintr>
    #define DSL_DEMUXER_NEW(name, maxBranches) \
        std::shared_ptr<DemuxerBintr>(new DemuxerBintr(name, maxBranches))

    #define DSL_SPLITTER_PTR std::shared_ptr<SplitterBintr>
    #define DSL_SPLITTER_NEW(name) \
        std::shared_ptr<SplitterBintr>(new SplitterBintr(name))

    /**
     * @class TeeBintr
     * @brief Implements a virtual base Tee binter that can add, link, 
     * unlink, and remove child branches
     */
    class TeeBintr : public Bintr
    {
    public: 
    
        /**
         * @brief ctor for the MultiBranchesBintr
         * @param[in] name name to give the new Bintr
         */
        TeeBintr(const char* name) 
            : Bintr(name)
            , m_blockingTimeout(DSL_TEE_DEFAULT_BLOCKING_TIMEOUT_IN_SEC)
        {
            LOG_FUNC();
        };

        /**
         * @brief dtor for the MultiBranchesBintr
         */
        ~TeeBintr()
        {
            LOG_FUNC();
        };

        /**
         * @brief adds a child ComponentBintr to this TeeBintr
         * @param[in] pChildComponent shared pointer to ComponentBintr to add
         * @return true if the ComponentBintr was added correctly, false otherwise
         */
        virtual bool AddChild(DSL_BINTR_PTR pChildComponent) = 0;
        
        /**
         * @brief removes a child ComponentBintr from this TeeBintr
         * @param[in] pChildComponent a shared pointer to ComponentBintr to remove
         * @return true if the ComponentBintr was removed correctly, false otherwise
         */
        virtual bool RemoveChild(DSL_BINTR_PTR pChildComponent) = 0;

        /**
         * @brief overrides the base method and checks in m_pChildBranches only.
         */
        virtual bool IsChild(DSL_BINTR_PTR pChildComponent) = 0;

        /**
         * @brief overrides the base Noder method to only return the number of 
         * child ComponentBintrs and not the total number of children... 
         * i.e. exclude the nuber of child Elementrs from the count
         * @return the number of Child ComponentBintrs (branches) held by this 
         * TeeBintr
         */
        virtual uint GetNumChildren() = 0;

        uint GetBlockingTimeout()
        {
            LOG_FUNC();
            
            return m_blockingTimeout;
        };
     
        bool SetBlockingTimeout(uint timeout)
        {
            LOG_FUNC();
            
            if (IsLinked())
            {
                LOG_ERROR("Unable to set blocking-timeout for Tee '" << GetName() 
                    << "' as it's currently linked");
                return false;
            }
            m_blockingTimeout = timeout;
            
            return true;
        };
        
    protected:

        /**
         * @brief current blocking-timeout for this TeeBintr. Controls
         * how long the TeeBintr will block/wait for the blocking PPH
         * callback to be called. Needs to be > seconds/frame for all streams.
         */
        uint m_blockingTimeout;
    
    };

    /**
     * @class MultiBranchesBintr
     * @brief Implements a base Tee binter that can add, link, unlink, and remove
     * child branches while in any state (NULL, PLAYING, etc.)
     */
    class MultiBranchesBintr : public TeeBintr
    {
    public: 
    
        /**
         * @brief ctor for the MultiBranchesBintr
         * @param[in] name name to give the new Bintr
         */
        MultiBranchesBintr(const char* name, const char* teeType);

        /**
         * @brief dtor for the MultiBranchesBintr
         */
        ~MultiBranchesBintr();

        /**
         * @brief adds a child ComponentBintr to this MultiBranchesBintr
         * @param[in] pChildComponent shared pointer to ComponentBintr to add
         * @return true if the ComponentBintr was added correctly, false otherwise
         */
        bool AddChild(DSL_BINTR_PTR pChildComponent);
        
        /**
         * @brief removes a child ComponentBintr from this MultiBranchesBintr
         * @param[in] pChildComponent a shared pointer to ComponentBintr to remove
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
         * @brief links all child Component Bintrs and their elements
         */ 
        bool LinkAll();
        
        /**
         * @brief unlinks all child Component Bintrs and their Elementrs
         */
        void UnlinkAll();
        
        /**
         * @brief sets the batch size for this Bintr
         * @param[in] batchSize the new batch size to use
         */
        bool SetBatchSize(uint batchSize);
        
    protected:
    
        /**
         * @brief Each source is assigned a unique request pad id when linked
         * the vector is used on dynamic add/remove to find the next available
         * stream id.
         */
        std::vector<bool> m_usedRequestPadIds;
    
        /**
         * @brief container of all child Sinks/Branches mapped by their unique names
         */
        std::map<std::string, DSL_BINTR_PTR> m_pChildBranches;

        /**
         * @brief container of all child sources mapped by their unique stream-id
         */
        std::map<uint, DSL_BINTR_PTR> m_pChildBranchesIndexed;

        /**
         * @brief adds a child Elementr to this PipelineSourcesBintr
         * @param pChildElement a shared pointer to the Elementr to add
         * @return a shared pointer to the Elementr if added correctly, 
         * nullptr otherwise
         */
        bool AddChild(DSL_BASE_PTR pChildElement);
        
        /**
         * @brief removes a child Elementr from this MultiBranchesBintr
         * @param pChildElement a shared pointer to the Elementr to remove
         */
        bool RemoveChild(DSL_BASE_PTR pChildElement);

        /**
         * @brief queue element for this Bintr to create a new process 
         * for this Bintr to run in
         */
        DSL_ELEMENT_PTR m_pQueue;
        
        /**
         * @brief Tee element -- multi-sinks, splitter or demuxer i.e. the
         * actual plugin is specific to the derived child class below.
         */
        DSL_ELEMENT_PTR m_pTee;
        

    };

    //-------------------------------------------------------------------------------

    /**
     * @class MultiSinksBintr
     * @brief Derived from the parent class MultiBranchesBintr, implements 
     * a Tee binter that can add, link, unlink, and remove child SinkBintrs 
     * while in any state (NULL, PLAYING, etc.)
     */
    class MultiSinksBintr : public MultiBranchesBintr
    {
    public: 
    
        /**
         * @brief ctor for the MultiSinksBintr
         * @param[in] name name to give the new Bintr
         */
        MultiSinksBintr(const char* name);

    };

    //-------------------------------------------------------------------------------

    class SplitterBintr : public MultiBranchesBintr
    {
    public: 
    
        /**
         * @brief ctor for the MultiSinksBintr
         * @param[in] name name to give the new Bintr
         */
        SplitterBintr(const char* name);

        /**
         * @brief Adds the SplitterBintr to a Parent Pipeline/Branch Bintr
         * @param[in] pParentBintr Parent Pipeline/Branch to add this Bintr to
         */
        bool AddToParent(DSL_BASE_PTR pParentBintr);

    };

    //-------------------------------------------------------------------------------
    
    class DemuxerBintr : public MultiBranchesBintr
    {
    public: 
    
        /**
         * @brief ctor for the DemuxerBintr
         * @param[in] name name to give the new Bintr
         * @param[in] maxBranches the maximum number of branches that can be
         * added/connected to this Demuxer, before or during Pipeline play.
         */
        DemuxerBintr(const char* name, uint maxBranches);
        
        /**
         * @brief Adds the DemuxerBintr to a Parent Branch/Pipeline Bintr
         * @param[in] pParentBintr Parent Branch/Pipeline to add this Bintr to
         */
        bool AddToParent(DSL_BASE_PTR pParentBintr);

        /**
         * @brief Adds a child ComponentBintr to this DemuxerBintr. We need to 
         * override the parent class because we pre-allocate the requested pads.
         * This is a workaround for the NIVIDA demuxer limatation of not allowing
         * pads to be requested in a PLAYING state.
         * @param[in] pChildComponent shared pointer to ComponentBintr to add
         * @return true if the ComponentBintr was added correctly, false otherwise
         */
        bool AddChild(DSL_BINTR_PTR pChildComponent);
        
        /**
         * @brief Adds a child ComponentBintr to this DemuxerBintr to a specified
         * stream-id and Demuxer source pd.
         * @param[in] pChildComponent shared pointer to ComponentBintr to add
         * @param[in] stream_id the stream-id and demuxer source pad to link to.
         * @return true if the ComponentBintr was added correctly, false otherwise
         */
        bool AddChildTo(DSL_BINTR_PTR pChildComponent, uint streamId);
        
        /**
         * @brief Moves a child ComponentBintr, owned by this DemuxerBintr, from its
         * current stream to a new stream.
         * stream-id and Demuxer source pd.
         * @param[in] pChildComponent shared pointer to ComponentBintr to move
         * @param[in] stream_id the destination stream-id to move to.
         * @return true if the ComponentBintr was moved correctly, false otherwise
         */
        bool MoveChildTo(DSL_BINTR_PTR pChildComponent, uint streamId);
        
        /** 
         * @brief links all child Component Bintrs and their elements. We need to 
         * override the parent class because we pre-allocate the requested pads.
         * This is a workaround for the NIVIDA demuxer limatation of not allowing
         * pads to be requested in a PLAYING state.
         */ 
        bool LinkAll();

        /**
         * @brief unlinks all child Component Bintrs and their Elementrs.
         */
        void UnlinkAll();
        
        /**
         * @brief Gets the current max-branches setting for this DemuxerBintr
         * @return current max-branches setting
         */
        uint GetMaxBranches();
        
        /**
         * @brief Set the max-branches setting for this DemuxerBintr.
         * @param[in] maxBranches the maximum number of branches that can be
         * added/connected to this Demuxer, before or during Pipeline play.
         * @return 
         */
        bool SetMaxBranches(uint maxBranches);

    private:

        /**
         * @brief Common code to complete the add child ComponentBintr process.
         * @param[in] pChildComponent shared pointer to ComponentBintr to add
         * @param[in] stream_id the stream-id and demuxer source pad to link to.
         * @return true if the ComponentBintr was added correctly, false otherwise
         */
        bool _completeAddChild(DSL_BINTR_PTR pChildComponent, uint streamId);
    
        /**
         * @brief maximum number of branches this DemuxerBintr can connect.
         * Specifies the number of source pads to request prior to playing.
         */
        uint m_maxBranches;
        
        /**
         * @brief list of reguest pads -- maxBranches in length -- for the 
         * for the DemuxerBintr. The pads are preallocated on Bintr LinkAll
         * and released on UnlinkAll.
         */
        std::vector<GstPad*> m_requestedSrcPads;
        
    };

}

#endif // _DSL_PROCESS_BINTR_H