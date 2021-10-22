/*
The MIT License

Copyright (c)   2021, Prominence AI, Inc.

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
#include "DslApi.h"
#include "DslServices.h"
#include "DslServicesValidate.h"
#include "DslSinkWebRtcBintr.h"

namespace DSL
{
    DslReturnType Services::SinkWebRtcNew(const char* name, const char* stunServer, 
        const char* turnServer, uint codec, uint bitrate, uint interval)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            // ensure component name uniqueness 
            if (m_components.find(name) != m_components.end())
            {   
                LOG_ERROR("Sink name '" << name << "' is not unique");
                return DSL_RESULT_SINK_NAME_NOT_UNIQUE;
            }
            if (codec != DSL_CODEC_H264 and codec != DSL_CODEC_H265)
            {   
                LOG_ERROR("Invalid Codec value = " << codec 
                    << " for WebRTC Sink '" << name << "'");
                return DSL_RESULT_SINK_CODEC_VALUE_INVALID;
            }
            m_components[name] = DSL_WEBRTC_SINK_NEW(name,
                stunServer, turnServer, codec, bitrate, interval);

            LOG_INFO("New WebRTC Sink '" << name << "' created successfully");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("New WebRTC Sink '" << name << "' threw exception on create");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkWebRtcServersGet(const char* name,
        const char** stunServer, const char** turnServer)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, WebRtcSinkBintr);

            DSL_WEBRTC_SINK_PTR pWebRtcSinkBintr = 
                std::dynamic_pointer_cast<WebRtcSinkBintr>(m_components[name]);

            pWebRtcSinkBintr->GetServers(stunServer, turnServer);

            LOG_INFO("STUN Sever = " << *stunServer << " TURN Server = " << *turnServer << 
                " returned successfully for WebRTC Sink '" << name << "'");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("WebRTC Sink '" << name << "' threw an exception getting dimensions");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkWebRtcServersSet(const char* name,
        const char* stunServer, const char* turnServer)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, WebRtcSinkBintr);

            DSL_WEBRTC_SINK_PTR pWebRtcSinkBintr = 
                std::dynamic_pointer_cast<WebRtcSinkBintr>(m_components[name]);

            pWebRtcSinkBintr->SetServers(stunServer, turnServer);

            LOG_INFO("STUN Sever = " << *stunServer << " TURN Server = " << *turnServer << 
                " returned successfully for WebRTC Sink '" << name << "'");

            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("WebRTC Sink '" << name << "' threw an exception getting dimensions");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

    DslReturnType Services::SinkWebRtcClientListenerAdd(const char* name,
        dsl_sink_webrtc_client_listener_cb listener, void* clientData)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, WebRtcSinkBintr);

            DSL_WEBRTC_SINK_PTR pWebRtcSinkBintr = 
                std::dynamic_pointer_cast<WebRtcSinkBintr>(m_components[name]);

            if (!pWebRtcSinkBintr->AddClientListener(listener, clientData))
            {
                LOG_ERROR("WebRTC Sink '" << name 
                    << "' failed to add Client Listener");
                return DSL_RESULT_SINK_WEBRTC_CLIENT_LISTENER_ADD_FAILED;
            }
            LOG_INFO("Web RTC Sink '" << name 
                << "' added a Client Listener successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("WebRTC Sink '" << name 
                << "' threw an exception adding Client Event Listner");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }
    
    DslReturnType Services::SinkWebRtcClientListenerRemove(const char* name,
        dsl_sink_webrtc_client_listener_cb listener)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_servicesMutex);

        try
        {
            DSL_RETURN_IF_COMPONENT_NAME_NOT_FOUND(m_components, name);
            DSL_RETURN_IF_COMPONENT_IS_NOT_CORRECT_TYPE(m_components, name, WebRtcSinkBintr);

            DSL_WEBRTC_SINK_PTR pWebRtcSinkBintr = 
                std::dynamic_pointer_cast<WebRtcSinkBintr>(m_components[name]);

            if (!pWebRtcSinkBintr->RemoveClientListener(listener))
            {
                LOG_ERROR("WebRTC Sink '" << name 
                    << "' failed to remove client Listener");
                return DSL_RESULT_SINK_WEBRTC_CLIENT_LISTENER_REMOVE_FAILED;
            }
            LOG_INFO("WebRtc Sink '" << name 
                << "' removed a Client Listener successfully");
                
            return DSL_RESULT_SUCCESS;
        }
        catch(...)
        {
            LOG_ERROR("WebRTC Sink '" << name 
                << "' threw an exception removing Client Listner");
            return DSL_RESULT_SINK_THREW_EXCEPTION;
        }
    }

}