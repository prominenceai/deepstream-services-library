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

#include "Dsl.h"
#include "DslPipelineBusMgr.h"

namespace DSL
{
    PipelineBusMgr::PipelineBusMgr(const GstObject* pGstPipeline)
        : m_pGstPipeline(pGstPipeline)
        , m_gstBusWatch(0)
        , m_errorNotificationTimerId(0)
    {
        LOG_FUNC();

        _initMaps();

        g_mutex_init(&m_busWatchMutex);
        g_mutex_init(&m_lastErrorMutex);
        
        GstBus* pGstBus = gst_pipeline_get_bus(GST_PIPELINE(m_pGstPipeline));

        // install the watch function for the message bus
        m_gstBusWatch = gst_bus_add_watch(pGstBus, bus_watch, this);
        gst_object_unref(pGstBus);
    }

    PipelineBusMgr::~PipelineBusMgr()
    {
        LOG_FUNC();
        
        g_mutex_clear(&m_busWatchMutex);
        g_mutex_clear(&m_lastErrorMutex);
    }

    bool PipelineBusMgr::AddStateChangeListener(dsl_state_change_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        
        if (m_stateChangeListeners.find(listener) != m_stateChangeListeners.end())
        {   
            LOG_ERROR("Pipeline listener is not unique");
            return false;
        }
        m_stateChangeListeners[listener] = clientData;
        
        return true;
    }

    bool PipelineBusMgr::RemoveStateChangeListener(dsl_state_change_listener_cb listener)
    {
        LOG_FUNC();
        
        if (m_stateChangeListeners.find(listener) == m_stateChangeListeners.end())
        {   
            LOG_ERROR("Pipeline listener was not found");
            return false;
        }
        m_stateChangeListeners.erase(listener);
        
        return true;
    }

    bool PipelineBusMgr::AddEosListener(dsl_eos_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        
        if (m_eosListeners.find(listener) != m_eosListeners.end())
        {   
            LOG_ERROR("Pipeline listener is not unique");
            return false;
        }
        m_eosListeners[listener] = clientData;
        
        return true;
    }

    bool PipelineBusMgr::RemoveEosListener(dsl_eos_listener_cb listener)
    {
        LOG_FUNC();
        
        if (m_eosListeners.find(listener) == m_eosListeners.end())
        {   
            LOG_ERROR("Pipeline listener was not found");
            return false;
        }
        m_eosListeners.erase(listener);
        
        return true;
    }

    bool PipelineBusMgr::AddErrorMessageHandler(dsl_error_message_handler_cb handler, void* clientData)
    {
        LOG_FUNC();
        
        if (m_errorMessageHandlers.find(handler) != m_errorMessageHandlers.end())
        {   
            LOG_ERROR("Pipeline handler is not unique");
            return false;
        }
        m_errorMessageHandlers[handler] = clientData;
        
        return true;
    }

    bool PipelineBusMgr::RemoveErrorMessageHandler(dsl_error_message_handler_cb handler)
    {
        LOG_FUNC();
        
        if (m_errorMessageHandlers.find(handler) == m_errorMessageHandlers.end())
        {   
            LOG_ERROR("Pipeline handler was not found");
            return false;
        }
        m_errorMessageHandlers.erase(handler);
        
        return true;
    }
    
    bool PipelineBusMgr::HandleBusWatchMessage(GstMessage* pMessage)
    {
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_busWatchMutex);
        
        switch (GST_MESSAGE_TYPE(pMessage))
        {
        case GST_MESSAGE_ELEMENT:
        case GST_MESSAGE_STREAM_STATUS:
        case GST_MESSAGE_DURATION_CHANGED:
        case GST_MESSAGE_QOS:
        case GST_MESSAGE_NEW_CLOCK:
        case GST_MESSAGE_ASYNC_DONE:
            LOG_INFO("Message type:: " 
                << gst_message_type_get_name(GST_MESSAGE_TYPE(pMessage)));
            break;
        case GST_MESSAGE_TAG:
            break;
        case GST_MESSAGE_EOS:
            HandleEosMessage(pMessage);
            break;
        case GST_MESSAGE_INFO:
            break;
        case GST_MESSAGE_WARNING:
            break;
        case GST_MESSAGE_ERROR:
            HandleErrorMessage(pMessage);            
            break;
        case GST_MESSAGE_STATE_CHANGED:
            HandleStateChanged(pMessage);
            break;
        default:
            LOG_INFO("Unhandled message type:: " 
                << gst_message_type_get_name(GST_MESSAGE_TYPE(pMessage)));
        }
        return true;
    }

    bool PipelineBusMgr::HandleStateChanged(GstMessage* pMessage)
    {
        if (GST_ELEMENT(GST_MESSAGE_SRC(pMessage)) != GST_ELEMENT(m_pGstPipeline))
        {
            return false;
        }

        GstState oldstate, newstate;
        gst_message_parse_state_changed(pMessage, &oldstate, &newstate, NULL);

        LOG_INFO(m_mapPipelineStates[oldstate] << " => " << m_mapPipelineStates[newstate]);

        // iterate through the map of state-change-listeners calling each
        for(auto const& imap: m_stateChangeListeners)
        {
            try
            {
                imap.first((uint)oldstate, (uint)newstate, imap.second);
            }
            catch(...)
            {
                LOG_ERROR("Exception calling Client State-Change-Lister");
            }
        }
        return true;
    }
    
    void PipelineBusMgr::HandleEosMessage(GstMessage* pMessage)
    {
        LOG_INFO("EOS message recieved");
        
        // iterate through the map of EOS-listeners calling each
        for(auto const& imap: m_eosListeners)
        {
            try
            {
                imap.first(imap.second);
            }
            catch(...)
            {
                LOG_ERROR("Exception calling Client EOS-Lister");
            }
        }
    }
    
    void PipelineBusMgr::HandleErrorMessage(GstMessage* pMessage)
    {
        LOG_FUNC();
        
        GError* error = NULL;
        gchar* debugInfo = NULL;
        gst_message_parse_error(pMessage, &error, &debugInfo);

        LOG_ERROR("Error message '" << error->message << "' received from '" 
            << GST_OBJECT_NAME(pMessage->src) << "'");
            
        if (debugInfo)
        {
            LOG_DEBUG("Debug info: " << debugInfo);
        }

        // persist the last error information
        std::string cstrSource(GST_OBJECT_NAME(pMessage->src));
        std::string cstrMessage(error->message);

        std::wstring wstrSource(cstrSource.begin(), cstrSource.end());
        std::wstring wstrMessage(cstrMessage.begin(), cstrMessage.end());
        
        // Setting the last error message will invoke a timer thread to notify all client handlers.
        SetLastErrorMessage(wstrSource, wstrMessage);
        
        g_error_free(error);
        g_free(debugInfo);
    }    

    void PipelineBusMgr::GetLastErrorMessage(std::wstring& source, std::wstring& message)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_lastErrorMutex);
        
        source = m_lastErrorSource;
        message = m_lastErrorMessage;
    }

    void PipelineBusMgr::SetLastErrorMessage(std::wstring& source, std::wstring& message)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_lastErrorMutex);

        m_lastErrorSource = source;
        m_lastErrorMessage = message;
        
        if (m_errorMessageHandlers.size())
        {
            m_errorNotificationTimerId = g_timeout_add(1, ErrorMessageHandlersNotificationHandler, this);
        }
    }
    
    int PipelineBusMgr::NotifyErrorMessageHandlers()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_lastErrorMutex);

        // iterate through the map of state-change-listeners calling each
        for(auto const& imap: m_errorMessageHandlers)
        {
            try
            {
                imap.first(m_lastErrorSource.c_str(), m_lastErrorMessage.c_str(), imap.second);
            }
            catch(...)
            {
                LOG_ERROR("PipelineBusMgr threw exception calling Client Error-Message-Handlers");
            }
        }
        // clear the timer id and return false to self remove
        m_errorNotificationTimerId = 0;
        return false;
    }

    void PipelineBusMgr::_initMaps()
    {
        m_mapPipelineStates[GST_STATE_READY] = "GST_STATE_READY";
        m_mapPipelineStates[GST_STATE_PLAYING] = "GST_STATE_PLAYING";
        m_mapPipelineStates[GST_STATE_PAUSED] = "GST_STATE_PAUSED";
        m_mapPipelineStates[GST_STATE_NULL] = "GST_STATE_NULL";
    }
    
    static gboolean bus_watch(GstBus* bus, GstMessage* pMessage, gpointer pData)
    {
        return static_cast<PipelineBusMgr*>(pData)->HandleBusWatchMessage(pMessage);
    }    
    
    static int ErrorMessageHandlersNotificationHandler(gpointer pPipeline)
    {
        return static_cast<PipelineBusMgr*>(pPipeline)->
            NotifyErrorMessageHandlers();
    }
    
} // DSL   