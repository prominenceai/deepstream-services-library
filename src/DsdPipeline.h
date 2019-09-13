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

#ifndef _DSD_PIPELINE_H
#define _DSD_PIPELINE_H

#include "DsdConfig.h"
#include "DsdBintr.h"

namespace DSD {

    /**
     * @class Pipline
     * @brief 
     */
    class Pipeline
    {
    public:
    
        /** 
         * 
         */
        Pipeline(Config& config, Display* pDisplay);
        ~Pipeline();
        
        bool Pause();
        bool Play();
        
        /**
         * @brief handles incoming Message Packets received
         * by the bus watcher callback function
         * @return true if the message was handled correctly 
         */
        bool HandleBusWatchMessage(GstMessage* pMessage);

        /**
         * @brief handles incoming sync messages
         * @param message incoming message to process
         * @return [GST_BUS_PASS|GST_BUS_FAIL]
         */
        GstBusSyncReply HandleBusSyncMessage(GstMessage* pMessage);
    
    private:

        /**
         * @brief underlying GStream pipeline wrapped by this class
         */
        GstElement* m_pPipeline;
        
        /**
         * @brief config object used by this pipeline
         */
        Config& m_config;
        
        /**
         * @brief handle to a common display used by all pipelines
         */
        Display* m_pDisplay;
        
        Bintr* m_pPipelineBintr;
        Bintr* m_pProcessingBintr;
        Bintr* m_pSinksBintr;
        Bintr* m_pOsdBintr;
        Bintr* m_pSourcesBintr;
        Bintr* m_pTrackerBintr;
        Bintr* m_pPrimaryGieBintr;
        Bintr* m_pSecondaryGiesBintr;
        Bintr* m_pTiledDisplayBintr;
        
        GstElement* m_pProcessingBin;


        /**
         * @brief mutex to protect critical pipeline code
         */
        GMutex m_pipelineMutex;

        /**
         * @brief mutex to prevent callback reentry
         */
        GMutex m_busWatchMutex;

        /**
         * @brief mutex to prevent callback reentry
         */
        GMutex m_busSyncMutex;

        /**
         * @brief Bus used to receive GstMessage packets.
         */
        GstBus* m_pGstBus;
        
        /**
         * @brief handle to the installed Bus Watch function.
         */
        guint m_gstBusWatch;
        
        NvDsSrcParentBin m_multiSourceBin;
        
        /**
         * @brief maps a GstState constant value to a string for logging
         */
        std::map<GstState, std::string> m_mapPipelineStates;
        
        /**
         * @brief maps a GstMessage constant value to a string for logging
         */
        std::map<GstMessageType, std::string> m_mapMessageTypes;

        bool HandleStateChanged(GstMessage* pMessage);
        /**
         * @brief initializes the "constant-value-to-string" maps
         */
        void _initMaps();
        
        
    }; // Pipeline
    
    /**
     * @brief callback function to watch a pipeline's bus for messages
     * @param bus instance pointer
     * @param message incoming message packet to process
     * @param pData pipeline instance pointer
     * @return true if the message was handled correctly 
     */
    static gboolean bus_watch(
        GstBus* bus, GstMessage* pMessage, gpointer pData);

    /**
     * @brief 
     * @param bus instance pointer
     * @param message incoming message packet to process
     * @param pData pipeline instance pointer
     * @return [GST_BUS_PASS|GST_BUS_FAIL]
     */
    static GstBusSyncReply bus_sync_handler(
        GstBus* bus, GstMessage* pMessage, gpointer pData);
        
    
} // Namespace

#endif // _DSD_PIPELINE_H

