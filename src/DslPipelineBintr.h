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

 * 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef _DSL_PIPELINE_H
#define _DSL_PIPELINE_H

#include "Dsl.h"
#include "DslApi.h"
#include "DslBranchBintr.h"
#include "DslPipelineStateMgr.h"
#include "DslPipelineXWinMgr.h"
#include "DslSourceBintr.h"
#include "DslDewarperBintr.h"
#include "DslPipelineSourcesBintr.h"
    
namespace DSL 
{
    /**
     * @brief convenience macros for shared pointer abstraction
     */
    #define DSL_PIPELINE_PTR std::shared_ptr<PipelineBintr>
    #define DSL_PIPELINE_NEW(name) \
        std::shared_ptr<PipelineBintr>(new PipelineBintr(name))

    /**
     * @class PipelineBintr
     * @brief 
     */
    class PipelineBintr : public BranchBintr, public PipelineStateMgr,
        public PipelineXWinMgr
    {
    public:
    
        /** 
         * 
         */
        PipelineBintr(const char* pipeline);
        ~PipelineBintr();
        
        /**
         * @brief Links all Child Bintrs owned by this Pipeline Bintr
         * @return True success, false otherwise
         */
        bool LinkAll();

        /**
         * @brief Attempts to link all and play the Pipeline
         * @return true if able to play, false otherwise
         */
        bool Play();

        /**
         * @brief Schedules a Timer Callback to call HandlePause in the mainloop context
         * @return true if HandlePause schedule correctly, false otherwise 
         */
        bool Pause();
        
        /**
         * @brief Pauses the Pipeline by setting its state to GST_STATE_PAUSED
         * Import: must be called in the mainloop's context, i.e. timer callback
         */
        void HandlePause();

        /**
         * @brief Schedules a Timer Callback to call HandleStop in the mainloop context
         * @return true if HandleStop schedule correctly, false otherwise 
         */
        bool Stop();
        
        /**
         * @brief Stops the Pipeline by setting its state to GST_STATE_NULL
         * Import: must be called in the mainloop's context, i.e. timer callback
         */
        void HandleStop();
        
        /**
         * @brief returns whether the Pipeline has all live sources or not.
         * @return true if all sources are live, false otherwise (default when no sources).
         */
        bool IsLive();

        /**
         * @brief adds a single Source Bintr to this Pipeline 
         * @param[in] pSourceBintr shared pointer to Source Bintr to add
         */
        bool AddSourceBintr(DSL_BASE_PTR pSourceBintr);

        bool IsSourceBintrChild(DSL_BASE_PTR pSourceBintr);

        /**
         * @brief returns the number of Sources currently in use by
         * this Pipeline
         */
        uint GetNumSourcesInUse()
        {
            if (!m_pPipelineSourcesBintr)
            {
                return 0;
            }
            return m_pPipelineSourcesBintr->GetNumChildren();
        } 
        
        /**
         * @brief removes a single Source Bintr from this Pipeline 
         * @param[in] pSourceBintr shared pointer to Source Bintr to add
         */
        bool RemoveSourceBintr(DSL_BASE_PTR pSourceBintr);

        /**
         * @brief Gets the current nvbuf memory type in use by the Stream-Muxer
         * @return one of DSL_NVBUF_MEM_TYPE constant values
         */
        uint GetStreamMuxNvbufMemType();

        /**
         * @brief Sets the nvbuf memory type for the Stream-Muxer to use
         * @param[in] type one of DSL_NVBUF_MEM_TYPE constant values
         * @return true if the memory type could be set, false otherwise
         */
        bool SetStreamMuxNvbufMemType(uint type);

        /**
         * @brief Gets the current batch settings for the Pipeline's Stream-Muxer
         * @param[out] batchSize current batchSize, default == the number of source
         * @param[out] batchTimeout current batch timeout
         * @return true if the batch properties could be read, false otherwise
         */
        void GetStreamMuxBatchProperties(uint* batchSize, uint* batchTimeout);

        /**
         * @brief Sets the current batch settings for the Pipeline's Stream-Muxer
         * @param[in] batchSize new batchSize to set, default == the number of sources
         * @param[in] batchTimeout timeout value to set in ms
         * @return true if the batch properties could be set, false otherwise
         */
        bool SetStreamMuxBatchProperties(uint batchSize, uint batchTimeout);

        /**
         * @brief Gets the current dimensions for the Pipeline's Stream Muxer
         * @param[out] width width in pixels for the current setting
         * @param[out] height height in pixels for the curren setting
         * @return true if the output dimensions could be read, false otherwise
         */
        bool GetStreamMuxDimensions(uint* width, uint* height);

        /**
         * @brief Set the dimensions for the Pipeline's Stream Muxer
         * @param width width in pixels to set the streamMux Output
         * @param height height in pixels to set the StreamMux output
         * @return true if the output dimensions could be set, false otherwise
         */
        bool SetStreamMuxDimensions(uint width, uint height);
        
        /**
         * @brief Gets the current setting for the Pipeline's Muxer padding
         * @param enable true if enabled, false otherwise.
         * @return true if the Padding enabled setting could be read, false otherwisee
         */
        bool GetStreamMuxPadding(bool* enabled);

        /**
         * @brief Sets, enables/disables the Pipeline's Stream Muxer padding
         * @param enabled set to true to enable padding
         * @return true if the Padding enabled setting could be set, false otherwise.
         */
        bool SetStreamMuxPadding(bool enabled);
        
        /**
         * @brief Gets the current setting for the Pipeline's StreamMuxer
         * num-surfaces-per-frame seting
         * @param[out] num current setting for the number of surfaces [1..4].
         * @return true if the number setting could be read, false otherwisee
         */
        bool GetStreamMuxNumSurfacesPerFrame(uint* num);

        /**
         * @brief Sets the current setting for the PipelineSourcesBintr's StreamMuxer
         * num-surfaces-per-frame seting
         * @param[in] num new value for the number of surfaces [1..4].
         * @return true if the number setting could be set, false otherwisee
         */
        bool SetStreamMuxNumSurfacesPerFrame(uint num);
        
        /**
         * @brief Adds a TilerBintr to be added to the Stream-muxers output
         * on link and play.
         * @param[in] pTilerBintr shared pointer to the tiler to add.
         * @return true if the Tiler was successfully added, false otherwise.
         */
        bool AddStreamMuxTiler(DSL_BASE_PTR pTilerBintr);
        
        /**
         * @brief Removes a TilerBintr previously added with AddStreamMuxTiler.
         * @return true if the TileBintr was successfully removed, false otherwise.
         */
        bool RemoveStreamMuxTiler();
        
        /**
         * @brief dumps a Pipeline's graph to dot file.
         * @param[in] filename name of the file without extention.
         * The caller is responsible for providing a correctly formated filename
         * The diretory location is specified by the GStreamer debug 
         * environment variable GST_DEBUG_DUMP_DOT_DIR
         */ 
        void DumpToDot(char* filename);
        
        /**
         * @brief dumps a Pipeline's graph to dot file prefixed
         * with the current timestamp.  
         * @param[in] filename name of the file without extention.
         * The caller is responsible for providing a correctly formated filename
         * The diretory location is specified by the GStreamer debug 
         * environment variable GST_DEBUG_DUMP_DOT_DIR
         */ 
        void DumpToDotWithTs(char* filename);
        
    private:

        /**
         * @brief Mutex to protect the async GCond used to synchronize
         * the Application thread with the mainloop context on
         * asynchronous change of pipeline state.
         */
        GMutex m_asyncCommMutex;
        
        /**
         * @brief Condition used to block the application context while waiting
         * for a Pipeline change of state to be completed in the mainloop context
         */
        GCond m_asyncCondition;
        
        /**
         * @brief parent bin for all Source bins in this Pipeline
         */
        DSL_PIPELINE_SOURCES_PTR m_pPipelineSourcesBintr;
        
        
        /**
         * @brief optional Tiler for the Stream-muxer's output
         */
        DSL_TILER_PTR m_pStreamMuxTilerBintr;
        
        
    }; // Pipeline
    
    /**
     * @brief Timer callback function to Stop a Pipeline in the mainloop context.  
     * @param pPipeline shared pointer to the Pipeline that started the timer to 
     * schedule the stop
     * @return false always to self destroy the on-shot timer.
     */
    static int PipelineStop(gpointer pPipeline);
    
} // Namespace

#endif // _DSL_PIPELINE_H

