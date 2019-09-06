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
 
    Pipeline::Pipeline()
        : m_pPipeline(NULL)
        , m_pGstBus(NULL)
        , m_gstBusWatch(0)
        , m_multiSourceBin{0}
    {
        LOG_FUNC();
        
        m_pPipeline = gst_pipeline_new("pipeline");
        
        if (!m_pPipeline)
        {
            LOG_ERROR("Failed to create new pipeline");
            throw;
        }

        g_mutex_init(&m_pipelineMutex);
        g_mutex_init(&m_busSyncMutex);
        g_mutex_init(&m_busWatchMutex);

        m_mapPipelineStates[GST_STATE_READY] = "GST_STATE_READY";
        m_mapPipelineStates[GST_STATE_PLAYING] = "GST_STATE_PLAYING";
        m_mapPipelineStates[GST_STATE_PAUSED] = "GST_STATE_PAUSED";
        m_mapPipelineStates[GST_STATE_NULL] = "GST_STATE_NULL";
        
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
        
        switch (GST_MESSAGE_TYPE(pMessage))
        {
        case GST_MESSAGE_INFO:
            LOG_INFO("Info message received");
            break;
        case GST_MESSAGE_WARNING:
            LOG_WARN("Warning message received");
            break;
        case GST_MESSAGE_ERROR:
            LOG_ERROR("Error message received");
            break;
        case GST_MESSAGE_STATE_CHANGED:
            return HandleStateChanged(pMessage);
        case GST_STATE_NULL:
            LOG_INFO("State NULL received");
            break;
        default:
            LOG_INFO("Unhandled message type:: " << GST_MESSAGE_TYPE(pMessage));
            break;
        }
        return true;
    }

    GstBusSyncReply Pipeline::HandleBusSyncMessage(GstMessage* pMessage)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_busSyncMutex);

        switch (GST_MESSAGE_TYPE(pMessage))
        {
        case GST_MESSAGE_ELEMENT:
            LOG_INFO("Processing message element");
            
            if (GST_MESSAGE_SRC(pMessage) == GST_OBJECT(m_multiSourceBin.bin))
            {
                const GstStructure *structure;
                structure = gst_message_get_structure(pMessage);
            }
            break;
        case GST_MESSAGE_STATE_CHANGED:
            LOG_INFO("State changed message received");
            break;
        default:
            LOG_INFO("Unhandled message type:: " << GST_MESSAGE_TYPE(pMessage));
            break;
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