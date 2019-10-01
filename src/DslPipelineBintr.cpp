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

#include "Dsl.h"
#include "DslPipelineBintr.h"

#include <gst/gst.h>

namespace DSL
{
    PipelineBintr::PipelineBintr(const char* pipeline)
        : Bintr(pipeline)
        , m_pGstPipeline(NULL)
        , m_pGstBus(NULL)
        , m_gstBusWatch(0)
    {
        LOG_FUNC();

        m_pGstPipeline = gst_pipeline_new((gchar*)pipeline);
        if (!m_pGstPipeline)
        {
            LOG_ERROR("Failed to create new GST Pipeline for '" << pipeline << "'");
            throw;
        }
        
        m_pSourcesBintr = std::shared_ptr<SourcesBintr>(new SourcesBintr("sources-bin"));
        m_pProcessBintr = std::shared_ptr<Bintr>(new Bintr("process-bin"));
        
        m_pSinksBintr = std::shared_ptr<Bintr>(new SinksBintr("sinks-bin"));
        
        AddChild(m_pSourcesBintr);
        AddChild(m_pProcessBintr);
        AddChild(m_pSinksBintr);
        
        // Initialize "constant-to-string" maps
        _initMaps();
        
        g_mutex_init(&m_pipelineMutex);
        g_mutex_init(&m_busSyncMutex);
        g_mutex_init(&m_busWatchMutex);

        // get the GST message bus - one per GST pipeline
        m_pGstBus = gst_pipeline_get_bus(GST_PIPELINE(m_pGstPipeline));
        
        // install the watch function for the message bus
        m_gstBusWatch = gst_bus_add_watch(m_pGstBus, bus_watch, this);
        
        // install the sync handler for the message bus
        gst_bus_set_sync_handler(m_pGstBus, bus_sync_handler, this, NULL);        
    }

    PipelineBintr::~PipelineBintr()
    {
        LOG_FUNC();

        gst_element_set_state(m_pBin, GST_STATE_NULL);
        
        // cleanup all resources
        gst_object_unref(m_pGstBus);

        g_mutex_clear(&m_pipelineMutex);
        g_mutex_clear(&m_busSyncMutex);
        g_mutex_clear(&m_busWatchMutex);
    }
    
    void PipelineBintr::AddSourceBintr(std::shared_ptr<Bintr> pBintr)
    {
        LOG_FUNC();
        
        m_pSourcesBintr->AddChild(pBintr);
    }

    void PipelineBintr::AddSinkBintr(std::shared_ptr<Bintr> pChildBintr)
    {
        LOG_FUNC();
        
        m_pSinksBintr->AddChild(pChildBintr);
    }

    void PipelineBintr::AddOsdBintr(std::shared_ptr<Bintr> pBintr)
    {
        LOG_FUNC();
        
        m_pOsdBintr = pBintr;
        
        AddChild(pBintr);
    }

    void PipelineBintr::AddPrimaryGieBintr(std::shared_ptr<Bintr> pGieBintr)
    {
        LOG_FUNC();
        
        if (m_pPrimaryGieBintr)
        {
            LOG_ERROR("Pipeline '" << m_name << "' has an exisiting Primary GIE '" << m_pPrimaryGieBintr->m_name);
            throw;
        }
        m_pPrimaryGieBintr = pGieBintr;
        
        AddChild(pGieBintr);
    }

    void PipelineBintr::AddDisplayBintr(std::shared_ptr<Bintr> pDisplayBintr)
    {
        m_pDisplayBintr = pDisplayBintr;
        
        AddChild(pDisplayBintr);
    }

    void PipelineBintr::SetStreamMuxProperties(gboolean areSourcesLive, 
        guint batchSize, guint batchTimeout, guint width, guint height)
    {
        
        LOG_FUNC();
    
        m_pSourcesBintr->SetStreamMuxProperties(areSourcesLive, 
            batchSize, batchTimeout, width, height);
    }

    bool PipelineBintr::Pause()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_pipelineMutex);
        
        return (gst_element_set_state(m_pBin, 
            GST_STATE_PAUSED) != GST_STATE_CHANGE_FAILURE);
    }

    bool PipelineBintr::Play()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_pipelineMutex);
                
        return (gst_element_set_state(m_pBin, 
            GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE);
    }
    
    bool PipelineBintr::HandleBusWatchMessage(GstMessage* pMessage)
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

    GstBusSyncReply PipelineBintr::HandleBusSyncMessage(GstMessage* pMessage)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_busSyncMutex);

        LOG_INFO(m_mapMessageTypes[GST_MESSAGE_TYPE(pMessage)] << " received");

        switch (GST_MESSAGE_TYPE(pMessage))
        {
        case GST_MESSAGE_ELEMENT:
            LOG_INFO("Processing message element");

            // Change to sources bin.
            if (GST_MESSAGE_SRC(pMessage) == GST_OBJECT(m_pBin))
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
    
    bool PipelineBintr::HandleStateChanged(GstMessage* pMessage)
    {
        LOG_FUNC();

        if (GST_ELEMENT(GST_MESSAGE_SRC(pMessage)) != m_pBin)
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
    
    void PipelineBintr::_initMaps()
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
        return static_cast<PipelineBintr *>(pData)->HandleBusWatchMessage(pMessage);
    }    
    
    static GstBusSyncReply bus_sync_handler(
        GstBus* bus, GstMessage* pMessage, gpointer pData)
    {
        return static_cast<PipelineBintr *>(pData)->HandleBusSyncMessage(pMessage);
    }
    

} // DSL