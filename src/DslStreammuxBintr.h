/*
The MIT License

Copyright (c) 2019-2024, Prominence AI, Inc.

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

#ifndef _DSL_STREAMMUX_BINTR_H
#define _DSL_STREAMMUX_BINTR_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslSourceBintr.h"

namespace DSL
{

    #define DSL_STREAMMUX_PTR std::shared_ptr<StreammuxBintr>
    #define DSL_STREAMMUX_NEW(name, parentBin, uniquePipelineId, srcPadName) \
        std::shared_ptr<StreammuxBintr> \
           (new StreammuxBintr(name, parentBin, uniquePipelineId, srcPadName))

    /**
     * @class StreammuxBintr
     * @brief Implements a Pipeline Streammux class for audio and video.
     */
    class StreammuxBintr : public Bintr
    {
    public: 
    
        /**
         * @brief ctor for the StreammuxBintr class
         * @param[in] name unique name for the StreammuxBintr to creat.
         * @param[in] parentBin gst bin pointer for the parent PipelineSourcesBintr.
         * @param[in] uniquePipelineId unique id for the Parent Pipeline, used to 
         * offset all source Id's (if greater than 0).
         * @param[in] ghostPadName name to assign the src ghost pad for the streammux.
         */
        StreammuxBintr(const char* name, 
            GstObject* parentBin, uint uniquePipelineId, const char* ghostPadName);

        /**
         * @brief dtor for the StreammuxBintr class.
         */
        ~StreammuxBintr();

        /**
         * @brief & operator for the StreammuxBintr class.
         * @return returns shared pointer to the Streammux elementr.
         */
        DSL_ELEMENT_PTR Get()
        {
            return m_pStreammux;
        }

        /**
         * @brief required by all Bintrs - single element, nothing to link.
         * @returns true on successful link, false otherwise.
         */
        bool LinkAll();
        
        /**
         * @brief required by all Bintrs - single element, nothing to link.
         */
        void UnlinkAll();

        /**
         * @brief Gets the current Streammuxer "play-type-is-live" setting.
         * @return true if play-type is live, false otherwise.
         */
        bool PlayTypeIsLiveGet();

        /**
         * @brief Sets the current Streammuxer play type based on the first source added.
         * @param isLive set to true if all sources are to be Live, and therefore live only.
         * @return true if live-source is succesfully set, false otherwise.
         */
        bool PlayTypeIsLiveSet(bool isLive);

        /**
         * @brief Gets the current config-file in use by the StreammuxBintr.
         * Default = NULL. Streammuxer will use all default vaules.
         * @return Current config file in use.
         */
        const char* GetConfigFile();
        
        /**
         * @brief Sets the config-file for the StreammuxBintr to use.
         * Default = NULL. Streammuxer will use all default vaules.
         * @param[in] configFile absolute or relative pathspec to new Config file.
         * @return True if the config-file property could be set, false otherwise.
         */
        bool SetConfigFile(const char* configFile);

        /**
         * @brief Gets the current batch size for the StreammuxBintr.
         * @return Current batchSize, default == the number of source.
         */
        uint GetBatchSize();

        /**
         * @brief Sets the current batch size for the StreammuxBintr.
         * @param[in] batchSize new batchSize to use.
         * @return true if batch-size is succesfully set, false otherwise.
         */
        bool SetBatchSize(uint batchSize);

        /**
         * @brief Gets the current setting for the StreammuxBintr's 
         * num-surfaces-per-frame seting
         * @return current setting for the number of surfaces [1..4].
         */
        uint GetNumSurfacesPerFrame();

        /**
         * @brief Sets the current setting for the StreammuxBintr's 
         * num-surfaces-per-frame seting.
         * @param[in] num new value for the number of surfaces [1..4].
         * @return true if dimensions are succesfully set, false otherwise.
         */
        bool SetNumSurfacesPerFrame(uint num);
        
        /**
         * @brief Gets the current setting for the StreammuxBintr's 
         * sync-inputs enabled property.
         * @preturn true if enabled, false otherwise.
         */
        boolean GetSyncInputsEnabled();
        
        /**
         * @brief Sets the StreammuxBintr's sync-inputs enabled property.
         * @param enabled set to true to enable sync-inputs, false otherwise.
         * @return true if sync-inputs enabled was succesfully set, false otherwise.
         */
        bool SetSyncInputsEnabled(boolean enabled);
        
        /**
         * @brief Gets the current setting for the StreammuxBintr's 
         * attach-sys-ts enabled property.
         * @preturn true if attach-sys-ts is enabled, false otherwise.
         */
        boolean GetAttachSysTsEnabled();
        
        /**
         * @brief Sets the StreammuxBintr's attach-sys-ts enabled property.
         * @param enabled set to true to enable attach-sys-ts, false otherwise.
         * @return true if attach-sys-ts enabled was succesfully set, false otherwise.
         */
        bool SetAttachSysTsEnabled(boolean enabled);
        
        /**
         * @brief Gets the current setting for the StreammuxBintr's 
         * max-latency property.
         * @preturn The maximum upstream latency in nanoseconds. 
         * When sync-inputs=1, buffers coming in after max-latency shall be dropped.
         */
        uint GetMaxLatency();
        
        /**
         * @brief Sets the StreammuxBintr's max-latency property.
         * @param[in] maxLatency the maximum upstream latency in nanoseconds. 
         * When sync-inputs=1, buffers coming in after max-latency shall be dropped.
         * @return true if max-latency was succesfully set, false otherwise.
         */
        bool SetMaxLatency(uint maxLatency);
        
        /** 
         * @brief Returns the state of the USE_NEW_NVSTREAMMUX env var.
         * @return true if USE_NEW_NVSTREAMMUX=yes, false otherwise.
         */
        bool UseNewStreammux(){return m_useNewStreammux;};

        //----------------------------------------------------------------------------
        // OLD NVSTREAMMUX SERVICES - Start
        
        /**
         * @brief Gets the current batch settings in use by the StreammuxBintr.
         * @param[out] batchSize current batchSize, default == the number of source.
         * @param[out] batchTimeout current batch timeout. Default = -1, disabled.
         */
        void GetBatchProperties(uint* batchSize, int* batchTimeout);

        /**
         * @brief Sets the current batch settings for the StreammuxBintr to use.
         * @param[in] batchSize new batchSize to set, default == the number of sources.
         * @param[in] batchTimeout timeout value to set in ms. Set to -1 to disable.
         * @return true if batch-properties are succesfully set, false otherwise.
         */
        bool SetBatchProperties(uint batchSize, int batchTimeout);

        /**
         * @brief Gets the StreammuxBintr's current NVIDIA buffer memory type.
         * @return one of the DSL_NVBUF_MEM_TYPE constant values.
         */
        uint GetNvbufMemType();

        /**
         * @brief Sets the StreammuxBintr's NVIDIA buffer memory type.
         * @param[in] type one of the DSL_NVBUF_MEM_TYPE constant values.
         * @return true if nvbuf-memory-type is succesfully set, false otherwise
         */
        bool SetNvbufMemType(uint type);

        /**
         * @brief Sets the StreammuxBintr's GPU setting.
         * @return true if successfully set, false otherwise.
         */
        bool SetGpuId(uint gpuId);
        
        /**
         * @brief Gets the StreammuxBintr's current dimensions.
         * @param[out] width width in pixels for the current setting.
         * @param[out] height height in pixels for the curren setting.
         */
        void GetDimensions(uint* width, uint* height);

        /**
         * @brief Set the StreammuxBintr's dimensions.
         * @param width width in pixels to set the streamMux Output.
         * @param height height in pixels to set the Streammux output.
         * @return true if dimensions are succesfully set, false otherwise.
         */
        bool SetDimensions(uint width, uint height);
        
        /**
         * @brief Gets the StreammuxBintr's padding enabled property.
         * @preturn true if enabled, false otherwise.
         */
        boolean GetPaddingEnabled();

        /**
         * @brief Sets the StreammuxBintr's padding enabled property.
         * @param enabled set to true to enable padding, false otherwise.
         * @return true if padding enabled was succesfully set, false otherwise.
         */
        bool SetPaddingEnabled(boolean enabled);

        /**
         * @brief true if batch-size explicity set by client, false by default.
         */
        bool m_batchSizeSetByClient;
         
        /**
         * @brief Each source is assigned a unique pad/stream id used to define the
         * streammuxer sink pad when linking. The vector is used on add/remove 
         * to find the next available pad id.
         */
        std::vector<bool> m_usedRequestPadIds;
        
    private:
    
        /**
         * @brief boolean flag to indicate if USE_NEW_NVSTREAMMUX=yes
         */
        bool m_useNewStreammux;

        /**
         * @brief unique id for the Parent Pipeline, used to offset all source
         * Id's (if greater than 0)
         */
        uint m_uniquePipelineId; 
        
        /**
         * @brief Pad Probe Event Handler to handle all dowstream nvstreammux
         * custom events [GST_NVEVENT_PAD_ADDED, GST_NVEVENT_PAD_DELETED,
         * GST_NVEVENT_STREAM_EOS, GST_NVEVENT_STREAM_SEGMENT]
         */
        DSL_PPEH_STREAM_EVENT_PTR m_pEventHandler;
        
        /**
         * @brief Pad Probe Handler to add the source-id offset (based on unique 
         * pipeline-id) for this StreammuxBintr
         */
        DSL_PPH_SOURCE_ID_OFFSETTER_PTR m_pSourceIdOffsetter;

        /**
         * @brief NVIDIA Streammux element for this StreammuxBintr.
         */
        DSL_ELEMENT_PTR m_pStreammux;

        /**
         * @brief true if all sources are live, false if all sources are non-live
         */
        bool m_areSourcesLive;
        
        /**
         * @brief Absolute or relative path to the Streammuxer config file.
         */
        std::string m_streammuxConfigFile;

        /**
         * @brief Number of surfaces-per-frame stream-muxer setting
         */
        int m_numSurfacesPerFrame;

        /**
         * @brief Compute Scaling HW to use. Applicable only for Jetson.
         * 0 (Default): Default, GPU for Tesla, VIC for Jetson
         * 1 (GPU): GPU
         * 2 (VIC): VIC
         */
        uint m_computeHw;
        
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

        // ---------------------------------------------------------------------------
        // OLD STREAMMUX PROPERTIES
        
        /**
         * @brief Stream-muxer batch timeout used when waiting for all sources
         * to produce a frame when batching together
         */
        gint m_batchTimeout;
        
        /**
         * @brief Stream-muxer batched frame output width in pixels
         */
        gint m_streamMuxWidth;

        /**
         * @brief Stream-muxer batched frame output height in pixels
         */
        gint m_streamMuxHeight;

        /**
         * @brief true if frame padding is enabled, false otherwise
         */
        boolean m_isPaddingEnabled;

        /**
         * @brief Number of buffers in output buffer pool
         */
        uint m_bufferPoolSize;

    };

}

#endif // _DSL_STREAMMUX_BINTR_H
