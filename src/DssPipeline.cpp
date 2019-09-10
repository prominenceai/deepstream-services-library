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

#include "Dss.h"
#include "DssPipeline.h"

#include <gst/gst.h>

namespace DSS
{
     Pipeline::Pipeline(Config& config, Display* pDisplay)
        : m_config(config)
        , m_pDisplay(pDisplay)
        , m_pPipeline(NULL)
        , m_pGstBus(NULL)
        , m_gstBusWatch(0)
        , m_pProcessingBin(NULL)
        , m_pPipelineBintr(NULL)
        , m_pProcessingBintr(NULL)
        , m_pSinksBintr(NULL)
        , m_pOsdBintr(NULL)
        , m_pSourcesBintr(NULL)
        , m_pTrackerBintr(NULL)
        , m_pPrimaryGieBintr(NULL)
        , m_pSecondaryGiesBintr(NULL)
        , m_pTiledDisplayBintr(NULL)
    {
        LOG_FUNC();
        
        // New GST Pipeline
        m_pPipeline = gst_pipeline_new("pipeline");
        if (!m_pPipeline)
        {
            LOG_ERROR("Failed to create new pipeline");
            throw;
        }

        // New processing parent bin 
        m_pProcessingBin = gst_bin_new("processing-bin");
        if (!m_pProcessingBin)
        {
            LOG_ERROR("Failed to create new processing-bin");
            throw;
        }
        
        m_pPipelineBintr = new Bintr("pipeline", (GstElement*)m_pPipeline);
        m_pProcessingBintr = new Bintr("processing", m_pProcessingBin);

        //  BUILD PIPELINE

        // Starting with the multi source bin
        m_pSourcesBintr = new Bintr("sources", m_config.CreateSourcesBin());
        m_pPipelineBintr->AddChild(m_pSourcesBintr);
        
            
        // Create the child procing bins for SINKS and OSD
        m_pSinksBintr = new Bintr("sinks", m_config.CreateSinksBin());
        m_pOsdBintr = new Bintr("osd", m_config.CreateOsdBin());

        // Add both to the Processing Bintr
        m_pProcessingBintr->AddChild(m_pSinksBintr);
        m_pProcessingBintr->AddChild(m_pOsdBintr);
            
        // Link the OSD and SINKS bins
        m_pOsdBintr->Link(m_pSinksBintr);
        
        // Add the proccessing bin to the Pipeline
        m_pPipelineBintr->AddChild(m_pProcessingBintr);

        // Create the Tiled Display bin and add it to Pipeline
        m_pTiledDisplayBintr = new Bintr("tiled-display", 
            m_config.CreateTiledDisplayBin());
        m_pPipelineBintr->AddChild(m_pTiledDisplayBintr);

        // Link the Tiled Display and the processing bin
//        m_pTiledDisplayBintr->Link(m_pProcessingBintr);

        // COMMON ELEMNTS
        
        // Create the Secondary GIEs bin and add it to Pipeline
        m_pSecondaryGiesBintr = new Bintr("secondary-gies", 
            m_config.CreateSecondaryGiesBin());
        m_pPipelineBintr->AddChild(m_pSecondaryGiesBintr);

        // Create the Tracker bin and add it to the Pipeline
//        m_pTrackerBintr = new Bintr("tracker", m_config.CreateTrackerBin());
//        m_pPipelineBintr->AddChild(m_pTrackerBintr);

        // Create the Primary GIE bin and add it to Pipeline
        m_pPrimaryGieBintr = new Bintr("primary-gie", 
            m_config.CreatePrimaryGieBin());
        m_pPipelineBintr->AddChild(m_pPrimaryGieBintr);
        
        m_pPrimaryGieBintr->Link(m_pSecondaryGiesBintr);


        // Initialize "constant-to-string" maps
        _initMaps();
        
        m_config.ConfigureTiledDisplay();

        g_mutex_init(&m_pipelineMutex);
        g_mutex_init(&m_busSyncMutex);
        g_mutex_init(&m_busWatchMutex);

        // get the GST message bus - one per GST pipeline
        m_pGstBus = gst_pipeline_get_bus(GST_PIPELINE(m_pPipeline));
        
        // install the watch function for the message bus
        m_gstBusWatch = gst_bus_add_watch(m_pGstBus, bus_watch, this);
        
        // install the sync handler for the message bus
        gst_bus_set_sync_handler(m_pGstBus, bus_sync_handler, this, NULL);        
    }

    Pipeline::~Pipeline()
    {
        LOG_FUNC();

        gst_element_set_state(m_pPipeline, GST_STATE_NULL);
        
        // cleanup all resources
        gst_object_unref(m_pGstBus);
        gst_object_unref(m_pPipeline);

        g_mutex_clear(&m_pipelineMutex);
        g_mutex_clear(&m_busSyncMutex);
        g_mutex_clear(&m_busWatchMutex);
    }
    
    bool Pipeline::Pause()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_pipelineMutex);
        
        return (gst_element_set_state(m_pPipeline, 
            GST_STATE_PAUSED) != GST_STATE_CHANGE_FAILURE);
    }

    bool Pipeline::Play()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_pipelineMutex);
        
        return (gst_element_set_state(m_pPipeline, 
            GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE);
    }
    
    bool Pipeline::HandleBusWatchMessage(GstMessage* pMessage)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_busWatchMutex);
        
        LOG_INFO(m_mapMessageTypes[GST_MESSAGE_TYPE(pMessage)] << " received");
        
        switch (GST_MESSAGE_TYPE(pMessage))
        {
        case GST_MESSAGE_INFO:
            return true;
        case GST_MESSAGE_WARNING:
            return true;
        case GST_MESSAGE_ERROR:
            return true;
        case GST_MESSAGE_STATE_CHANGED:
            return HandleStateChanged(pMessage);
        case GST_MESSAGE_EOS:
          return FALSE;
      break;
        default:
            LOG_INFO("Unhandled message type:: " << m_mapMessageTypes[GST_MESSAGE_TYPE(pMessage)]);
        }
        return true;
    }

    GstBusSyncReply Pipeline::HandleBusSyncMessage(GstMessage* pMessage)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_busSyncMutex);

        LOG_INFO(m_mapMessageTypes[GST_MESSAGE_TYPE(pMessage)] << " received");

        switch (GST_MESSAGE_TYPE(pMessage))
        {
        case GST_MESSAGE_ELEMENT:
            LOG_INFO("Processing message element");
            
            if (GST_MESSAGE_SRC(pMessage) == GST_OBJECT(m_multiSourceBin.bin))
            {
                const GstStructure *structure;
                structure = gst_message_get_structure(pMessage);
            }
            return GST_BUS_PASS;
        case GST_MESSAGE_STATE_CHANGED:
            return GST_BUS_PASS;

        default:
            LOG_INFO("Unhandled message type:: " << m_mapMessageTypes[GST_MESSAGE_TYPE(pMessage)]);
        }
        return GST_BUS_PASS;
    }
    
    bool Pipeline::HandleStateChanged(GstMessage* pMessage)
    {
        LOG_FUNC();

        if (GST_ELEMENT(GST_MESSAGE_SRC(pMessage)) != m_pPipeline)
        {
            return false;
        }

        GstState oldstate, newstate;

        gst_message_parse_state_changed(pMessage, &oldstate, &newstate, NULL);
        
        LOG_INFO(m_mapPipelineStates[oldstate] << " => " << m_mapPipelineStates[newstate]);
            
//        switch(newstate)
//        {
//        case GST_STATE_NULL:
//        case GST_STATE_PLAYING:
//        case GST_STATE_READY:
//        case GST_STATE_PAUSED:
//        default:
//        }
        return true;
    }
    
    void Pipeline::_initMaps()
    {
        m_mapMessageTypes[GST_MESSAGE_UNKNOWN] = "GST_MESSAGE_UNKNOWN";
        m_mapMessageTypes[GST_MESSAGE_EOS] = "GST_MESSAGE_EOS";
        m_mapMessageTypes[GST_MESSAGE_INFO] = "GST_MESSAGE_INFO";
        m_mapMessageTypes[GST_MESSAGE_WARNING] = "GST_MESSAGE_WARNING";
        m_mapMessageTypes[GST_MESSAGE_ERROR] = "GST_MESSAGE_ERROR";
        m_mapMessageTypes[GST_MESSAGE_TAG] = "GST_MESSAGE_TAG";
        m_mapMessageTypes[GST_MESSAGE_BUFFERING] = "GST_MESSAGE_BUFFERING";
        m_mapMessageTypes[GST_MESSAGE_STATE_CHANGED] = "GST_MESSAGE_STATE_CHANGED";
        m_mapMessageTypes[GST_MESSAGE_STEP_DONE] = "GST_MESSAGE_STEP_DONE";
        m_mapMessageTypes[GST_MESSAGE_CLOCK_LOST] = "GST_MESSAGE_CLOCK_LOST";
        m_mapMessageTypes[GST_MESSAGE_NEW_CLOCK] = "GST_MESSAGE_NEW_CLOCK";
        m_mapMessageTypes[GST_MESSAGE_STREAM_STATUS] = "GST_MESSAGE_STREAM_STATUS";

        m_mapPipelineStates[GST_STATE_READY] = "GST_STATE_READY";
        m_mapPipelineStates[GST_STATE_PLAYING] = "GST_STATE_PLAYING";
        m_mapPipelineStates[GST_STATE_PAUSED] = "GST_STATE_PAUSED";
        m_mapPipelineStates[GST_STATE_NULL] = "GST_STATE_NULL";
    }

    static gboolean bus_watch(
        GstBus* bus, GstMessage* pMessage, gpointer pData)
    {
        return static_cast<Pipeline *>(pData)->HandleBusWatchMessage(pMessage);
    }    
    
    static GstBusSyncReply bus_sync_handler(
        GstBus* bus, GstMessage* pMessage, gpointer pData)
    {
        return static_cast<Pipeline *>(pData)->HandleBusSyncMessage(pMessage);
    }
    

} // DSS