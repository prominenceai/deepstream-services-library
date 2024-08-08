/*
The MIT License

Copyright (c) 2023-2-24, Prominence AI, Inc.

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
#include "DslQBintr.h"

namespace DSL
{
    
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_REMUXER_BRANCH_PTR std::shared_ptr<RemuxerBranchBintr>
    #define DSL_REMUXER_BRANCH_NEW(name, \
        parentRemuxerBin, pChildBranch, streamIds, numStreamIds) \
        std::shared_ptr<RemuxerBranchBintr>(new RemuxerBranchBintr(name, \
            parentRemuxerBin, pChildBranch, streamIds, numStreamIds))

    #define DSL_REMUXER_PTR std::shared_ptr<RemuxerBintr>
    #define DSL_REMUXER_NEW(name) \
        std::shared_ptr<RemuxerBintr>(new RemuxerBintr(name))

    /**
     * @class RemuxerBranchBintr
     * @brief Implements a Remuxer-Branch Proxy Bintr
     */
    class RemuxerBranchBintr : public Bintr
    {
    public: 
    
        /**
         * @brief ctor for the RemuxerBranchBintr class
         * @param[in] name name to give the new RemuxerBranchBintr
         * @param[in] parentRemuxerBin the GstObj for the Parent RemuxerBintr.
         * @param[in] pChildBranch child Branch for this RemuxerBranchBintr.
         * @param[in] streamIds array of stream-ids identifying the streams to
         * connect this RemuxerBranchBintr
         * @param[in] numStreamIds number of stream-ids in the streamIds array.
         */
        RemuxerBranchBintr(const char* name, GstObject* parentRemuxerBin, 
            DSL_BINTR_PTR pChildBranch, uint* streamIds, uint numStreamIds);
            
        /**
         * @brief dtor for the RemuxerBranchBintr class
         */
        ~RemuxerBranchBintr();

        /**
         * @brief Links all child components of this RemuxerBranchBintr
         * @return true if all components were succesfully linked, false otherwise.
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all child components of the RemuxerBintr
         */
        void UnlinkAll();
        
        /**
         * @brief Links the RemuxerBranchBintr to the Parent RemuxerBintr's
         * collection of tees, one for each streamId in m_streamIds.
         * @param splitters[in] vector of tees to link to.
         * @return true if all streams were successfully linked
         */
        bool LinkToSourceTees(const std::vector<DSL_ELEMENT_PTR>& tees);
        
        /**
         * @brief Unlinks the RemuxerBranchBintr from the Parent RemuxerBintr's
         * collection of tees.
         */
        void UnlinkFromSourceTees();

        /**
         * @brief links this Noder to the Sink Pad of Muxer
         * @param[in] pMuxer nodetr to link to
         * @param[in] padName name to give the requested Sink Pad
         * @return true if able to successfully link with Muxer Sink Pad
         */
        bool LinkToSinkMuxer(DSL_NODETR_PTR pMuxer, const char* padName);

        /**
         * @brief unlinks this Nodetr from a previously linked Muxer Sink Pad
         * @return true if able to successfully unlink from Muxer Sink Pad
         */
        bool UnlinkFromSinkMuxer();
        
        /**
         * @brief Gets the Metamuxer branch config-string with Infer Id and
         * semicolon delimited list of Stream-Ids if specified.
         * @return Branch config-string if select streams, empty string otherwise.
         */
        std::string GetBranchConfigString(){return m_branchConfigString;};
        
        /**
         * @brief Gets the current batch settings for the RemuxerBranchBintr's 
         * Streammuxer.
         * @return current batch-size, default == the number of source.
         */
        uint GetBatchSize();

        /**
         * @brief Sets the batch-size for the RemuxerBranchBintr's streammuxer. 
         * @param[in] batchSize new batchSize to set, default == the number of sources.
         * @return true if batch-properties are succesfully set, false otherwise.
         */
        bool SetBatchSize(uint batchSize);

        /**
         * @brief Gets the current config-file in use by the Pipeline's Streammuxer.
         * Default = NULL. Streammuxer will use all default vaules.
         * @return Current config file in use.
         */
        const char* GetStreammuxConfigFile();
        
        /**
         * @brief Sets the config-file for the Pipeline's Streammuxer to use.
         * Default = NULL. Streammuxer will use all default vaules.
         * @param[in] configFile absolute or relative pathspec to new Config file.
         * @return True if the config-file property could be set, false otherwise,
         */
        bool SetStreammuxConfigFile(const char* configFile);

        /**
         * @brief Gets the current batch settings for the RemuxerBranchBintr's 
         * Streammuxer.
         * @param[out] batchSize current batchSize, default == the number of source.
         * @param[out] batchTimeout current batch timeout. Default = -1, disabled.
         */
        void GetBatchProperties(uint* batchSize, int* batchTimeout);

        /**
         * @brief Sets the current batch settings for the RemuxerBranchBintr's 
         * er.
         * @param[in] batchSize new batchSize to set, default == the number of sources.
         * @param[in] batchTimeout timeout value to set in ms. Set to -1 to disable.
         * @return true if batch-properties are succesfully set, false otherwise.
         */
        bool SetBatchProperties(uint batchSize, int batchTimeout);

        /**
         * @brief Gets the current dimensions for the RemuxerBranchBintr's 
         * Streammuxer.
         * @param[out] width width in pixels for the current setting
         * @param[out] height height in pixels for the curren setting
         */
        void GetDimensions(uint* width, uint* height);

        /**
         * @brief Set the dimensions for the RemuxerBranchBintr's Streammuxer
         * @param width width in pixels to set the streammux Output
         * @param height height in pixels to set the Streammux output
         * @return true if the output dimensions could be set, false otherwise
         */
        bool SetDimensions(uint width, uint height);

        /**
         * @brief Sets the NVIDIA buffer memory type for the RemuxerBranchBintr's 
         * Streammuxer.
         * @brief nvbufMemType new memory type to use, one of the 
         * DSL_NVBUF_MEM_TYPE constant values.
         * @return true if successfully set, false otherwise.
         */
        bool SetNvbufMemType(uint nvbufMemType);

        /**
         * @brief Sets the GPU ID for the RemuxerBranchBintr's Streammuxer.
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);
        
        /** 
         * @brief Returns the state of the USE_NEW_NVSTREAMMUX env var.
         * @return true if USE_NEW_NVSTREAMMUX=yes, false otherwise.
         */
        bool UseNewStreammux(){return m_useNewStreammux;};
        
    private:
    
        /**
         * @brief Child Streammuxer for the RemuxerBranchBintr
         */
        DSL_ELEMENT_PTR m_pStreammux;

        /**
         * @brief boolean flag to indicate if USE_NEW_NVSTREAMMUX=yes
         */
        bool m_useNewStreammux;

        /**
         * @brief Stream-muxer batch timeout used when waiting for all sources
         * to produce a frame when batching together
         */
        int m_batchTimeout;

        /**
         * @brief Streammuxer batched frame output width in pixels
         */
        uint m_width;

        /**
         * @brief Streammuxer batched frame output height in pixels
         */
        uint m_height;
        
        /**
         * @brief Child Branch to link to the Streammuxer
         */
        DSL_BINTR_PTR m_pChildBranch;
        
        /**
         * @brief Vector of stream-ids to connect this RemuxerBranchBintr to
         */
        std::vector<uint> m_streamIds;
        
        /**
         * @brief String stream of comma delimeted stream-ids
         */
        std::string m_branchConfigString;
        
        /**
         * @brief True if connecting to select stream-ids, false otherwise.
         */    
        bool m_linkSelectiveStreams;

        /**
         * @brief Absolute or relative path to the Streammuxer config file.
         */
        std::string m_streammuxConfigFile;

        /**
         * @brief Number of surfaces-per-frame stream-muxer setting
         */
        int m_numSurfacesPerFrame;

        /**
         * @brief Attach system timestamp as ntp timestamp, otherwise ntp 
         * timestamp calculated from RTCP sender reports.
         */
        boolean m_attachSysTs;
        
        /**
         * @brief if true, sychronizes input frames using PTS.
         */
        boolean m_syncInputs;
        
        /**
         * @brief The maximum upstream latency in nanoseconds. 
         * When sync-inputs=1, buffers coming in after max-latency shall be dropped.
         */
        uint m_maxLatency;

        /**
         * @brief Duration of input frames in milliseconds for use in NTP timestamp 
         * correction based on frame rate. If set to 0 (default), frame duration is 
         * inferred automatically from PTS values seen at RTP jitter buffer. When 
         * there is change in frame duration between the RTP jitter buffer and the 
         * nvstreammux, this property can be used to indicate the correct frame rate 
         * to the nvstreammux, for e.g. when there is an audiobuffersplit GstElement 
         * before nvstreammux in the pipeline. If set to -1, disables frame rate 
         * based NTP timestamp correction. 
         */
        int64_t m_frameDuration;

        /**
         * @brief property to control EOS propagation downstream from nvstreammux
         * when all the sink pads are at EOS. (Experimental)
         */
        boolean m_dropPipelineEos;

        /**
         * @brief Container of Queues elements used to connect to Streammuxer.
         * The components are mapped by target stream-id
         */
        std::map<uint, DSL_ELEMENT_PTR> m_queues;
        
    };

    // -------------------------------------------------------------------------------

    /**
     * @class RemuxerConfigFile
     * @brief Utility class to create a Metamuxer Config file. 
     */
    class RemuxerConfigFile
    {
    public:
       
        /**
         * @brief ctor for the RemuxerConfigFile
         * @param[in] filePath - absolute or relative path to use for file createion.
         */
        RemuxerConfigFile(std::string filePath)
        {
            LOG_FUNC();
            
            m_ostream.open(filePath.c_str(), std::fstream::out | std::fstream::trunc);
            m_ostream << "[property]" << std::endl;
            m_ostream << "[group-0]" << std::endl;
        }
        
        /**
         * @brief Adds a Branch config-string to [group-0]
         * @param[in] configString Branch config-string to add to the config file
         */
        void AddBranchConfigString(const std::string& configString)
        {
            m_ostream << configString.c_str() << std::endl;            
        }
        
        /**
         * @brief Closes the file
         */
        void Close()
        {
            m_ostream.close();
        }

    private:

        /**
         * @brief File stream used to create the config file. 
         */
        std::fstream m_ostream;
    };
    
    // -------------------------------------------------------------------------------

    /**
     * @class RemuxerBintr
     * @brief Implements a Remuxer (demuxer-streammuxers-metamuxer) bin container
     */
    class RemuxerBintr : public QBintr
    {
    public: 
    
        /**
         * @brief ctor for the RemuxerBintr class
         * @param[in] name name to give the new RemuxerBintr
         */
        RemuxerBintr(const char* name);

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
         * to streams produced by, and tee'd off of, the Bintr's Demuxer.
         * @return true if the ComponentBintr was added correctly, false otherwise
         */
        bool AddChild(DSL_BINTR_PTR pChildBranch);

        /**
         * @brief Adds a child ComponentBintr to this RemuxerBintr. Each child 
         * (branch) is assigned a new Streammuxer. Each Streammuxer is connected
         * to multiple streams produced by, and tee'd off of, the Bintr's Demuxer.
         * @param[in] pChildBranch shared pointer to Bintr (branch) to add.
         * @param[in] streamIds array of streamIds - identifing which streams to 
         * connect-to.
         * @return true if the ComponentBintr was added correctly, false otherwise
         */
        bool AddChildTo(DSL_BINTR_PTR pChildBranch, 
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
            
            return m_childBranches.size();
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
         * @brief Overrides the base SetBatchSize() to set the batch size for 
         * this Bintr.
         * @param[in] batchSize the new batchSize to use.
         */
        bool SetBatchSize(uint batchSize);

        /**
         * @brief Gets the current batch settings for the RemuxerBintr.
         * @return Current batchSize, default == the number of sources, set by 
         * the parent once the Pipeline is playing, if not overridden.
         */
        uint GetBatchSize();

        /**
         * @brief Overrides the parent (branch) batchsize for the RemuxerBintr.
         * @param[in] batchSize new batchSize to set, default is set by parent.
         * @return true if batch-size is succesfully set, false otherwise.
         */
        bool OverrideBatchSize(uint batchSize);

        // ---------------------------------------------------------------------------
        // NEW STREAMMUX SERVICES - Start
        
        /**
         * @brief Gets the current config-file in use by the specified child branch.
         * Default = NULL. Streammuxer will use all default vaules.
         * @param[in] pChildComponent child branch to query.
         * @return Current config file in use.
         */
        const char* GetStreammuxConfigFile(DSL_BINTR_PTR pChildComponent);
        
        /**
         * @brief Sets the config-file for the Pipeline's Streammuxer to use.
         * Default = NULL. Streammuxer will use all default vaules.
         * @param[in] pChildComponent child branch to update.
         * @param[in] configFile absolute or relative pathspec to new Config file.
         * @return True if the config-file property could be set, false otherwise,
         */
        bool SetStreammuxConfigFile(DSL_BINTR_PTR pChildComponent,
            const char* configFile);

        // ---------------------------------------------------------------------------
        // NEW STREAMMUX SERVICES - End
        // ---------------------------------------------------------------------------
        // OLD STREAMMUX SERVICES - Start
        /**
         * @brief Gets the current batch settings for the RemuxerBintr.
         * @param[out] batchSize current batchSize, default == the number of source.
         * @param[out] batchTimeout current batch timeout. Default = -1, disabled.
         */
        void GetBatchProperties(uint* batchSize, int* batchTimeout);

        /**
         * @brief Sets the current batch settings for the RemuxerBintr.
         * @param[in] batchSize new batchSize to set, default is set by parent.
         * @param[in] batchTimeout timeout value to set in ms. Set to -1 to disable.
         * @return true if batch-properties are succesfully set, false otherwise.
         */
        bool SetBatchProperties(uint batchSize, int batchTimeout);

        /**
         * @brief Gets the current output frame dimensions for the RemuxerBintr.
         * Settings are used by all Streammuxers created for this RemuxerBintr.
         * @param[out] width width in pixels for the current setting.
         * @param[out] height height in pixels for the curren setting.
         */
        void GetDimensions(uint* width, uint* height);

        /**
         * @brief Set the frame dimensions for the RemuxerBintr to use.
         * Settings are used by all Streammuxers created for this RemuxerBintr.
         * @param width width in pixels to set the streamMux Output.
         * @param height height in pixels to set the Streammux output.
         * @return true if the output dimensions could be set, false otherwise.
         */
        bool SetDimensions(uint width, uint height);

        /**
         * @brief Sets the NVIDIA buffer memory type for all children of this
         * RemuxerBintr.
         * @brief nvbufMemType new memory type to use, one of the 
         * DSL_NVBUF_MEM_TYPE constant values.
         * @return true if successfully set, false otherwise.
         */
        bool SetNvbufMemType(uint nvbufMemType);

        /**
         * @brief Sets the GPU ID for all children of this RemuxerBintr.
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);
        

        /** 
         * @brief Returns the state of the USE_NEW_NVSTREAMMUX env var.
         * @return true if USE_NEW_NVSTREAMMUX=yes, false otherwise.
         */
        bool UseNewStreammux(){return m_useNewStreammux;};
        
    private:
    
        /**
         * @brief path to create and load the Metamuxer config file.
         */
        std::string m_configFilePath;

        /**
         * @brief boolean flag to indicate if USE_NEW_NVSTREAMMUX=yes
         */
        bool m_useNewStreammux;
    
        /**
         * @brief Maximum number of stream-ids that can be connected.
         */
        uint m_maxStreamIds;
        
        /**
         * @brief true if batch-size explicity set by client, false by default.
         */
        bool m_batchSizeSetByClient;

        /**
         * @brief Batch-timeout used for all branches, i.e. all Streammuxers 
         */
        int m_batchTimeout;

        /**
         * @brief list of reguest pads -- batch-size in length -- for the 
         * for the DemuxerBintr. 
         * and then used on LinkAll or AddChild when in a linked-state
         */
        std::vector<GstPad*> m_requestedSrcPads;

        /**
         * @brief Batched frame output width in pixels for all branches.
         */
        uint m_width;

        /**
         * @brief Batched frame output height in pixels for all branches.
         */
        uint m_height;
        
        /**
         * @brief Active sink pad which buffer will transfer to src pad
         */
        uint m_activePad;
        
        /**
         * @brief Input Tee for the RemuxerBintr.
         */
        DSL_ELEMENT_PTR m_pInputTee;
        
        /**
         * @brief Metamuxer for the RemuxerBintr.
         */
        DSL_ELEMENT_PTR m_pMetamuxer;
        
        /**
         * @brief Streamdemuxer input queue for the RemuxerBintr.
         */
        DSL_ELEMENT_PTR m_pDemuxerQueue;
        
        /**
         * @brief Streamdemuxer for the RemuxerBintr.
         */
        DSL_ELEMENT_PTR m_pDemuxer;
        
        /**
         * @brief Vector of Splitter Tees, one per maxStreamIds, for the RemuxerBintr.
         */
        std::vector<DSL_ELEMENT_PTR> m_tees;

        /**
         * @brief container of all child Sinks/Branches mapped by their unique names
         */
        std::map<std::string, DSL_REMUXER_BRANCH_PTR> m_childBranches;
    };
    
}

#endif // _DSL_REMUXER_BINTR_H