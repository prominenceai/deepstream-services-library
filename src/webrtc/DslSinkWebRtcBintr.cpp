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
#include "DslApi.h"
#include "DslSinkWebRtcBintr.h"
#include "DslBranchBintr.h"

namespace DSL
{
    WebRtcSinkBintr::WebRtcSinkBintr(const char* name, const char* stunServer, 
        const char* turnServer, uint encoder, uint bitrate, uint iframeInterval)
        : EncodeSinkBintr(name, encoder, bitrate, iframeInterval)
        , SignalingTransceiver()
        , m_pDataChannel(NULL)
        , m_stunServer(stunServer)
        , m_turnServer(turnServer)
        , m_completeClosedTimerId(0)
    {
        LOG_FUNC();

        std::string fakeSinkName = GetName() + "-fake-sink";
        m_pFakeSinkBintr = DSL_FAKE_SINK_NEW(fakeSinkName.c_str());

        switch (encoder)
        {
        case DSL_ENCODER_HW_H264 :
            m_pPayloader = DSL_ELEMENT_NEW("rtph264pay", name);
            break;
        case DSL_ENCODER_HW_H265 :
            m_pPayloader = DSL_ELEMENT_NEW("rtph265pay", name);
            break;
        default:
            LOG_ERROR("Invalid encoder = '" << encoder << "' for new WebRtcSinkBintr '" 
                << name << "'");
            throw;
        }
        m_pWebRtcCapsFilter = DSL_ELEMENT_EXT_NEW("capsfilter", name, "payloader");
        
        GstCaps* pCaps = gst_caps_from_string(
            "application/x-rtp,media=video,encoding-name=H264,payload=96");
        m_pWebRtcCapsFilter->SetAttribute("caps", pCaps);
        gst_caps_unref(pCaps);

        LOG_INFO("");
        LOG_INFO("Initial property values for WebRtcSinkBintr '" << name << "'");
        LOG_INFO("  stun-server        : " << m_stunServer);
        LOG_INFO("  turn-server        : " << m_turnServer); 
        LOG_INFO("  encoder            : " << m_encoder);
        if (m_bitrate)
        {
            LOG_INFO("  bitrate            : " << m_bitrate);
        }
        else
        {
            LOG_INFO("  bitrate            : " << m_defaultBitrate);
        }
        LOG_INFO("  iframe-interval    : " << m_iframeInterval);
        LOG_INFO("  converter-width    : " << m_width);
        LOG_INFO("  converter-height   : " << m_height);
        LOG_INFO("  sync               : " << m_sync);
        LOG_INFO("  async              : " << m_async);
        LOG_INFO("  max-lateness       : " << m_maxLateness);
        LOG_INFO("  qos                : " << m_qos);
        LOG_INFO("  enable-last-sample : " << m_enableLastSample);
        LOG_INFO("  queue              : " );
        LOG_INFO("    leaky            : " << m_leaky);
        LOG_INFO("    max-size         : ");
        LOG_INFO("      buffers        : " << m_maxSizeBuffers);
        LOG_INFO("      bytes          : " << m_maxSizeBytes);
        LOG_INFO("      time           : " << m_maxSizeTime);
        LOG_INFO("    min-threshold    : ");
        LOG_INFO("      buffers        : " << m_minThresholdBuffers);
        LOG_INFO("      bytes          : " << m_minThresholdBytes);
        LOG_INFO("      time           : " << m_minThresholdTime);
        
        AddChild(m_pPayloader);
        AddChild(m_pWebRtcCapsFilter);

        SoupServerMgr::GetMgr()->AddSignalingTransceiver(this);
    }

    WebRtcSinkBintr::~WebRtcSinkBintr()
    {
        LOG_FUNC();
    
        SoupServerMgr::GetMgr()->RemoveSignalingTransceiver(this);

        if (IsLinked())
        {    
            UnlinkAll();
        }
    }

    bool WebRtcSinkBintr::LinkAll()
    {
        LOG_FUNC();
        
        if (m_isLinked)
        {
            LOG_ERROR("WebRtcSinkBintr '" << GetName() << "' is already linked");
        }

        m_pWebRtcBin = DSL_ELEMENT_NEW("webrtcbin", GetCStrName());

        // Set the STUN and/or TURN server 
        if (m_stunServer.size())
        {
            m_pWebRtcBin->SetAttribute("stun-server", m_stunServer.c_str());
        }
        if (m_turnServer.size())
        {
            m_pWebRtcBin->SetAttribute("turn-server", m_turnServer.c_str());
        }

        g_signal_connect(m_pWebRtcBin->GetGstObject(), "pad-added",
            G_CALLBACK(on_pad_added_cb), (gpointer)this);
        g_signal_connect(m_pWebRtcBin->GetGstObject(), "pad-removed",
            G_CALLBACK(on_pad_removed_cb), (gpointer)this);
        g_signal_connect(m_pWebRtcBin->GetGstObject(), "no-more-pads",
            G_CALLBACK(on_no_more_pads_cb), (gpointer)this);
        g_signal_connect(m_pWebRtcBin->GetGstObject(), "on-negotiation-needed", 
            G_CALLBACK(on_negotiation_needed_cb), (gpointer)this);
        g_signal_connect(m_pWebRtcBin->GetGstObject(), "on-ice-candidate",
            G_CALLBACK(on_ice_candidate_cb), (gpointer)this);
        g_signal_connect(m_pWebRtcBin->GetGstObject(), "on-new-transceiver",
            G_CALLBACK(on_new_transceiver_cb), (gpointer)this);
        g_signal_connect(m_pWebRtcBin->GetGstObject(), "on-data-channel",
            G_CALLBACK(on_data_channel_cb), (gpointer)this);

        AddChild(m_pWebRtcBin);

        if (!LinkToCommon(m_pPayloader) or 
            !m_pPayloader->LinkToSink(m_pWebRtcCapsFilter) or
            !m_pWebRtcCapsFilter->LinkToSink(m_pWebRtcBin))
        {
            return false;
        }
        m_isLinked = true;
        return true;
    }
    
    void WebRtcSinkBintr::UnlinkAll()
    {
        LOG_FUNC();
        
        if (!m_isLinked)
        {
            LOG_ERROR("WebRtcSinkBintr '" << GetName() << "' is not linked");
            return;
        }
        UnlinkFromCommon();
        m_pPayloader->UnlinkFromSink();
        m_pWebRtcCapsFilter->UnlinkFromSink();

        // remove and delete the webrtcbin to be recreated on next Linkall
        RemoveChild(m_pWebRtcBin);
        m_pWebRtcBin = nullptr;

        m_isLinked = false;
    }

    bool WebRtcSinkBintr::AddToParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();

        // cast the pointer to a branch pointer
        DSL_BRANCH_PTR pParentBranchBintr = 
            std::dynamic_pointer_cast<BranchBintr>(pParentBintr);

        // add the Fake Sink to the Parent branch regardless of state
        if (!pParentBranchBintr->AddSinkBintr(
                std::dynamic_pointer_cast<SinkBintr>(m_pFakeSinkBintr)))
        {
            LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                << "' failed to add its FakeSinkBintr to the parent branch '" 
                << pParentBranchBintr->GetName() << "'");
            return false;
        }
        LOG_INFO("Fake Sink FOR WebRtcSinkBintr '" << GetName() 
            << "' was added to the parent branch '" 
            << pParentBranchBintr->GetName() << "' successfully");

        // get the current state of the branch and add the WebRTC Sink if playing or paused
        // GstState state;
        // pParentBranchBintr->GetState(state, 0);
        // if (state == GST_STATE_PLAYING)
        // {
            // add the this WebRtcSinkBintr now
            if (!pParentBranchBintr->AddSinkBintr(
                    std::dynamic_pointer_cast<SinkBintr>(shared_from_this())))
            {
                LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                    << "' failed to add its FakeSinkBintr to the parent branch");
                return false;
            }
        // }
        // Setup the current Parent Pipeline/Branch pointer
        m_pParentBintr = pParentBintr;
        return true;

    }

    bool WebRtcSinkBintr::RemoveFromParent(DSL_BASE_PTR pParentBintr)
    {
        LOG_FUNC();
        
        // cast the pointer to a branch pointer
        DSL_BRANCH_PTR pParentBranchBintr = 
            std::dynamic_pointer_cast<BranchBintr>(pParentBintr);

        if (!m_pFakeSinkBintr->IsParent(pParentBintr))
        {
            LOG_ERROR("Fake Sink owned by WebRtcSinkBintr '" << GetName() 
                << "' is not a child of the parent branch '" 
                << pParentBranchBintr->GetName() << "'");
            return false;
        }

        // Clear the current Parent Pipeline/Branch pointer now.
        m_pParentBintr = nullptr;

        // if this WebRtcSinkBintr has been added to the parent branch (on socket connect)
        if (IsParent(pParentBintr))
        {
            // remove 'this' WebRtcSinkBintr from the Parent Pipeline 
            if (!pParentBranchBintr->RemoveSinkBintr(
                std::dynamic_pointer_cast<SinkBintr>(shared_from_this())))
            {
                LOG_ERROR("WebRtcSinkBintr '" << GetName()
                    << "' faild to remove its Fake Sink form the parent branch '" 
                    << pParentBranchBintr->GetName() << "'");
                return false;
            }
        }
        return true;
    }

    void WebRtcSinkBintr::GetServers(const char** stunServer, const char** turnServer)
    {
        LOG_FUNC();

        *stunServer = m_stunServer.c_str();
        *turnServer = m_turnServer.c_str();
    }

    bool WebRtcSinkBintr::SetServers(const char* stunServer, const char* turnServer)
    {
        LOG_FUNC();

        if (IsLinked())
        {
            LOG_ERROR("Unable to set STUN and TURN Settings for WebRtcSinkBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_stunServer.assign(stunServer);
        m_turnServer.assign(turnServer);

        // Set the STUN or TURN server properties
        m_pWebRtcBin->SetAttribute("stun-server", m_stunServer.c_str());
        m_pWebRtcBin->SetAttribute("turn-server", m_turnServer.c_str());

        return true;
    }

    bool WebRtcSinkBintr::AddClientListener(dsl_sink_webrtc_client_listener_cb listener, 
        void* clientData)
    {
        LOG_FUNC();

        if (m_clientListeners.find(listener) != m_clientListeners.end())
        {   
            LOG_ERROR("Client listener is not unique");
            return false;
        }
        m_clientListeners[listener] = clientData;
        
        return true;
    }
    
    bool WebRtcSinkBintr::RemoveClientListener(dsl_sink_webrtc_client_listener_cb listener)
    {
        LOG_FUNC();

        if (m_clientListeners.find(listener) == m_clientListeners.end())
        {   
            LOG_ERROR("Client listener was not found");
            return false;
        }
        m_clientListeners.erase(listener);
        
        return true;
    }

    void WebRtcSinkBintr::SetConnection(SoupWebsocketConnection* pConnection)
    {
        LOG_FUNC();

        if (m_pConnection)
        {
            LOG_ERROR("The WebRtcSinkBintr '" << GetName() 
                << "' is already in a connected state.");
            return;
        }
        if (m_pParentBintr == nullptr)
        {
            LOG_ERROR("The WebRtcSinkBintr '" << GetName() 
                << "' is not a child of any parent branch or pipeline");
            return;
        }

        // Call the base/super class to persist the connection pointer.
        SignalingTransceiver::SetConnection(pConnection);

        // cast the pointer to a branch pointer
        DSL_BRANCH_PTR pParentBranchBintr = 
            std::dynamic_pointer_cast<BranchBintr>(m_pParentBintr);

        // get the current state of the branch and add the WebRTC Sink if playing
        GstState state;
        pParentBranchBintr->GetState(state, 0);

        if (state != GST_STATE_PLAYING and m_clientListeners.empty())
        {
            LOG_ERROR("The WebRtcSinkBintr '" << GetName() << "' is currently stopped \\\
                and without client liseners is unable to connect");
            return;
        }

        if (!IsInUse())
        {
            // add "this" WebRtcSinkBintr now
            if (!pParentBranchBintr->AddSinkBintr(
                    std::dynamic_pointer_cast<SinkBintr>(shared_from_this())))
            {
                LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                    << "' failed to add itself from the parent branch");
                return;
            }
        }

        // IMPORTANT: it is up to a client listener to start the pipeline
        // If we're not currently in a state of playing. 

        // notify all client listeners that a new Websocket connection
        // has been initiated by a remote client. 
        notifyClientListeners();
    }

    bool WebRtcSinkBintr::CloseConnection()
    {
        LOG_FUNC();

        if (!m_pConnection)
        {
            LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                << "' is not in a connected state");
            return false;
        }
        gst_webrtc_data_channel_close(m_pDataChannel);
        // Closing the connection with a close code of 0 and no data.
        soup_websocket_connection_close(m_pConnection, 0, NULL);
        ClearConnection();
        return true;
    }

    void WebRtcSinkBintr::OnClosed(SoupWebsocketConnection* pConnection)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_transceiverMutex);

        LOG_INFO("on-close called for WebRtcSinkBintr '" << GetName() <<"'");

        // Need to close the WebRTC data channel if open.
        if (m_pDataChannel)
        {
//            gst_webrtc_data_channel_close(m_pDataChannel);
        }
        m_pDataChannel = NULL;

        m_completeClosedTimerId = g_timeout_add(1, complete_on_closed_cb, this);
    }

    int WebRtcSinkBintr::CompleteOnClosed()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_transceiverMutex);

        // Call the base/super class to clear the connection
        SignalingTransceiver::OnClosed(m_pConnection);

        // cast the parent pointer to a branch pointer
        DSL_BRANCH_PTR pParentBranchBintr = 
            std::dynamic_pointer_cast<BranchBintr>(m_pParentBintr);

        // remove "this" WebRtcSinkBintr, to be added back in on next connection. 
        if (!pParentBranchBintr->RemoveSinkBintr(
            std::dynamic_pointer_cast<SinkBintr>(shared_from_this())))
        {
            LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                << "' failed to remove itself to the parent branch");
            return false;
        }
        // notify all client listeners that the Websocket has now closed
        notifyClientListeners();

        m_completeClosedTimerId = 0;

        // return false to destroy/unref the timer.
        return false;
    }


    void WebRtcSinkBintr::OnMessage(SoupWebsocketConnection* pConnection, 
        SoupWebsocketDataType dataType, GBytes* message)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_transceiverMutex);

        // Get the client webrtcbin based on the connection


        switch (dataType)
        {
            case SOUP_WEBSOCKET_DATA_BINARY:
                LOG_ERROR("WebRtcSinkBintr '" << GetName() << "' received unknown binary message, ignoring");
                g_bytes_unref(message);
                return;

            case SOUP_WEBSOCKET_DATA_TEXT:
                break;

            default:
                LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                    << "' received unknown data type, ignoring");
                g_bytes_unref(message);
                return;
        }

        LOG_INFO("message-received for WebRtcSinkBintr '" 
            << GetName() << "'");

        // Copy the message to a g-byte-array and unreference the message
        gsize size;
        gchar* data = (gchar*)g_bytes_unref_to_data(message, &size);

        // Copy and convert to NULL-terminated string and free the byte-array 
        gchar* dataString = g_strndup(data, size);
        g_free(data);

        // Load the message into the JSON parser
        if (!json_parser_load_from_data(m_pJsonParser, dataString, -1, NULL))
        {
            LOG_ERROR("WebRtcSinkBintr received unknown data type");
            g_free(dataString);
            return;
        }

        // data has been loaded into the parser, free the string now
        g_free(dataString);

        JsonNode* pRootJson = json_parser_get_root(m_pJsonParser);
        if (!JSON_NODE_HOLDS_OBJECT(pRootJson))
        {
            LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                << "' received a message a without a JSON Root");
            return;
        } 

        JsonObject* pRootJsonObject = json_node_get_object(pRootJson);
        if (!json_object_has_member(pRootJsonObject, "type")) 
        {
            LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                << "' received a message without a type memeber");
            return;
        }

        const gchar* typeString = json_object_get_string_member(pRootJsonObject, "type");
        if (!json_object_has_member(pRootJsonObject, "data")) 
        {
            LOG_ERROR("WebRtcSinkBintr '" 
                << GetName() << "' received a message without data");
            return;
        }

        JsonObject* pDataJsonObject = json_object_get_object_member(
                pRootJsonObject, "data");

        if (g_strcmp0(typeString, "sdp") == 0) 
        {
            if (!json_object_has_member(pDataJsonObject, "type")) 
            {
                LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                    << "' received a SDP message without type field");
                return;
            }

            const gchar* sdpTypeString = json_object_get_string_member(
                    pDataJsonObject, "type");
            if (g_strcmp0 (sdpTypeString, "answer") != 0) 
            {
                LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                    << "' expected SDP message without type 'answer' but received "
                    << sdpTypeString << "");
                return;
            }

            if (!json_object_has_member(pDataJsonObject, "sdp")) 
            {
                LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                    << "' received a SDP message without SDP string");
                return;
            }

            const gchar* sdpString = json_object_get_string_member(pDataJsonObject, "sdp");

            LOG_INFO("WebRtcSinkBintr '" << GetName() 
                << "' received SDP: " << sdpString);

            GstSDPMessage *sdp;
            if (gst_sdp_message_new(&sdp) != GST_SDP_OK)
            {
                LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                    << "' failed to create new SDP message");
                return;
            }

            int ret = gst_sdp_message_parse_buffer((guint8 *)sdpString, 
                strlen(sdpString), sdp);
            if (ret != GST_SDP_OK) 
            {
                LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                    << "' failed to parse SDP message");
                return;
            }

            GstWebRTCSessionDescription* answer = gst_webrtc_session_description_new(
                    GST_WEBRTC_SDP_TYPE_ANSWER, sdp);
            if (!answer)
            {
                LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                    << "' failed to create new webrtc session answer");
                return;
            }

            GstPromise* pPromise = gst_promise_new_with_change_func(on_remote_desc_set_cb, 
                m_pWebRtcBin->GetGstObject(), NULL);

            g_signal_emit_by_name(G_OBJECT(m_pWebRtcBin->GetGObject()), 
                "set-remote-description", answer, pPromise);    
            gst_webrtc_session_description_free(answer);

            // emit signal to create data channel on first pass only
            if(m_pDataChannel == NULL)
            {
                g_signal_emit_by_name(m_pWebRtcBin->GetGObject(), "create-data-channel", 
                    "channel", NULL, &m_pDataChannel);
                if (!m_pDataChannel)
                {
                    LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                        << "' failed to create data channel - returning");
                    return;
                }

                // With the data channel now setup, time to connect the signal handlers
                ConnectDataChannelSignals((GObject*)m_pDataChannel);
            }
        }
        else if (g_strcmp0(typeString, "ice") == 0) 
        {
            if (!json_object_has_member(pDataJsonObject, "sdpMLineIndex")) 
            {
                LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                    << "' received ICE message without mline index");
                return;
            }

            if (!json_object_has_member(pDataJsonObject, "candidate")) 
            {
                LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                    << "' received ICE message without ICE candidate string");
                return;
            }

            const gchar* candidateString = json_object_get_string_member(
                    pDataJsonObject, "candidate");
            guint mlineIndex = json_object_get_int_member(
                pDataJsonObject, "sdpMLineIndex");

            LOG_INFO("WebRtcSinkBintr '" << GetName() 
                << "' received ICE candidate with mline index: " << std::to_string(mlineIndex) 
                << "; candidate: " << candidateString);

            g_signal_emit_by_name(m_pWebRtcBin->GetGstObject(), "add-ice-candidate", mlineIndex, candidateString);
        }
        else
        {
            LOG_ERROR("WebRtcSinkBintr '" << GetName() << "' received unknown message type " << typeString 
                << ", returning");
        }
    }

    void WebRtcSinkBintr::OnNegotiationNeeded()
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_transceiverMutex);

        LOG_INFO("on-negotiation-needed called for WebRtcSinkBintr '" 
            << GetName() << "'");

        GstPromise* pPromise = gst_promise_new_with_change_func(
            on_offer_created_cb, (gpointer)this, NULL);
        g_signal_emit_by_name(m_pWebRtcBin->GetGstObject(), 
            "create-offer", NULL, pPromise);
    }

    void WebRtcSinkBintr::OnOfferCreated(GstPromise* pPromise)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_transceiverMutex);

        LOG_INFO("on-offer-created called for WebRtcSinkBintr '" 
            << GetName() << "'");

        GstStructure const* pReply = gst_promise_get_reply(pPromise);
        GstWebRTCSessionDescription *pOffer = NULL;
        gst_structure_get(pReply, "offer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &pOffer, NULL);
        gst_promise_unref(pPromise);

        GstPromise* localDescPromise = gst_promise_new_with_change_func(
                on_local_desc_set_cb, (gpointer)this, NULL);
        g_signal_emit_by_name(m_pWebRtcBin->GetGstObject(), 
            "set-local-description", pOffer, localDescPromise);

        gchar* sdpStr = gst_sdp_message_as_text(pOffer->sdp);

        JsonObject* sdpJson = json_object_new();
        json_object_set_string_member(sdpJson, "type", "sdp");

        JsonObject* sdpDataJson = json_object_new();
        json_object_set_string_member(sdpDataJson, "type", "offer");
        json_object_set_string_member(sdpDataJson, "sdp", sdpStr);
        json_object_set_object_member(sdpJson, "data", sdpDataJson);

        gchar* jsonStr = getStrFromJsonObj(sdpJson);
        json_object_unref(sdpJson);

        soup_websocket_connection_send_text(m_pConnection, jsonStr);
        g_free(jsonStr);
        g_free(sdpStr);

        gst_webrtc_session_description_free(pOffer);  
    }

    void WebRtcSinkBintr::OnIceCandidate(guint mLineIndex, gchar* candidate)
    
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_transceiverMutex);

        LOG_INFO("on-ice-candidate '" << candidate << "' received for WebRtcSinkBintr '" 
            << GetName() << "'");

        JsonObject* iceJson = json_object_new();
        json_object_set_string_member(iceJson, "type", "ice");

        JsonObject* iceDataJson = json_object_new();
        json_object_set_int_member(iceDataJson, "sdpMLineIndex", mLineIndex);
        json_object_set_string_member(iceDataJson, "candidate", candidate);
        json_object_set_object_member(iceJson, "data", iceDataJson);

        gchar* jsonStr = getStrFromJsonObj(iceJson);
        json_object_unref(iceJson);

        soup_websocket_connection_send_text(m_pConnection, jsonStr);
        g_free(jsonStr);
    }

    void WebRtcSinkBintr::OnLocalDescSet(GstPromise* pPromise)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_transceiverMutex);

        LOG_INFO("on-local-desc-set called for WebRtcSinkBintr '" 
            << GetName() << "'");

        GstStructure const *pReply = gst_promise_get_reply(pPromise);
        if (pReply != NULL)
        {
            gchar* replyStr = gst_structure_to_string(pReply);
            LOG_INFO("Reply for on-local-desc-set is '" << replyStr
                << "' for WebRtcSinkBintr '" << GetName());
            g_free(replyStr);
        }
        gst_promise_unref(pPromise);  
    }

    void WebRtcSinkBintr::OnRemoteDescSet(GstPromise* pPromise)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_transceiverMutex);

        LOG_INFO("on-remote-desc-set called for WebRtcSinkBintr '");

        GstStructure const *pReply = gst_promise_get_reply(pPromise);
        if (pReply != NULL)
        {
            gchar* replyStr = gst_structure_to_string(pReply);
            LOG_INFO("Reply for on-remote-desc-set is '" << replyStr
                << "' for WebRtcSinkBintr '" << GetName());
            g_free(replyStr);
        }
        gst_promise_unref(pPromise);  
    }

    void WebRtcSinkBintr::ConnectDataChannelSignals(GObject* dataChannel)
    {
        LOG_FUNC();

        LOG_INFO("Connecting data channel signals for WebRtcSinkBintr '" 
            << GetName() << "'");

        // Setup the RTP data channel signal handlers
        m_dataChannelOnErrorSignalHandlerId = g_signal_connect(dataChannel, "on-error", 
            G_CALLBACK(data_channel_on_error_cb), this);
        m_dataChannelOnOpenSignalHandlerId = g_signal_connect(dataChannel, "on-open", 
            G_CALLBACK(data_channel_on_open_cb), this);
        m_dataChannelOnCloseSignalHandlerId = g_signal_connect(dataChannel, "on-close", 
            G_CALLBACK(data_channel_on_close_cb), this);
        m_dataChannelOnMessageSignalHandlerId = g_signal_connect(dataChannel, "on-message-string", 
            G_CALLBACK(data_channel_on_message_string_cb), this);
    }

    void WebRtcSinkBintr::DataChannelOnOpen(GObject* pDataChannel)
    {
        LOG_FUNC();

        LOG_INFO("data-channel-on-open called for WebRtcSinkBintr '" << GetName() << "'");

        GstWebRTCDataChannel* pQualifedDataChannel = (GstWebRTCDataChannel*)pDataChannel;

        std::string confirmation("Data channel for WebRTC Sink '" 
            + GetName() + "' opened successfully");

        GBytes *bytes = g_bytes_new("data", strlen("data"));
        g_signal_emit_by_name(pQualifedDataChannel, "send-string", confirmation.c_str());
        g_signal_emit_by_name(pQualifedDataChannel, "send-data", bytes);
        g_bytes_unref(bytes);
    }

    void WebRtcSinkBintr::DataChannelOnClose(GObject* pDataChannel)
    {
        LOG_FUNC();

        LOG_INFO("data-channel-on-close called for WebRtcSinkBintr '" << GetName() << "'");

        if (m_dataChannelOnErrorSignalHandlerId)
        {
            g_signal_handler_disconnect(G_OBJECT(pDataChannel), m_dataChannelOnErrorSignalHandlerId);
            m_dataChannelOnErrorSignalHandlerId = 0;
        }
        if (m_dataChannelOnOpenSignalHandlerId)
        {
            g_signal_handler_disconnect(G_OBJECT(pDataChannel), m_dataChannelOnOpenSignalHandlerId);
            m_dataChannelOnOpenSignalHandlerId = 0;
        }
        if (m_dataChannelOnCloseSignalHandlerId)
        {
            g_signal_handler_disconnect(G_OBJECT(pDataChannel), m_dataChannelOnCloseSignalHandlerId);
            m_dataChannelOnCloseSignalHandlerId = 0;
        }
        if (m_dataChannelOnMessageSignalHandlerId)
        {
            g_signal_handler_disconnect(G_OBJECT(pDataChannel), m_dataChannelOnMessageSignalHandlerId);
            m_dataChannelOnOpenSignalHandlerId = 0;
        }
    }

    // ------------------------------------------------------------------------------
    // Private Member Functions

    void WebRtcSinkBintr::notifyClientListeners()
    {
        LOG_FUNC();

        // If we have registered client listeners for the WebRtSinkBintr
        if (m_clientListeners.size())
        {
            // setup the connection information data for the client listener(s)
            dsl_webrtc_connection_data data{0};

            // single field for now
            data.current_state = m_connectionState;

            // iterate through the map of client listeners calling each
            for(auto const& imap: m_clientListeners)
            {
                try
                {
                    imap.first(&data, imap.second);
                }
                catch(...)
                {
                    LOG_ERROR("Exception calling Web RTC Sink Client Listener");
                }
            }
        }
    }

    gchar* WebRtcSinkBintr::getStrFromJsonObj(JsonObject * object)
    {
        LOG_FUNC();

        /* Make it the root node */
        JsonNode* root = json_node_init_object(json_node_alloc (), object);
        JsonGenerator* generator = json_generator_new();
        json_generator_set_root(generator, root);
        gchar* text = json_generator_to_data(generator, NULL);

        /* Release everything */
        g_object_unref(generator);
        json_node_free(root);
        return text;
    }

    // -------------------------------------------------------------------------------
    // Signal Callback Functions

    static void on_pad_added_cb(GstElement* pWebrtcbin, GstPad* pad, gpointer pWebRtcSink)
    {
        LOG_INFO("on-pad-added called for WebRtcSinkBintr '"
            << static_cast<WebRtcSinkBintr*>(pWebRtcSink)->GetName() << "'");
    }

    static void on_pad_removed_cb(GstElement* pWebrtcbin, GstPad* pad, gpointer pWebRtcSink)
    {
        LOG_INFO("on-pad-removed called for WebRtcSinkBintr '" 
            << static_cast<WebRtcSinkBintr*>(pWebRtcSink)->GetName() << "'");
    }
 
    static void on_no_more_pads_cb(GstElement* pWebrtcbin, gpointer pWebRtcSink)
    {
        LOG_WARN("on-no-more-pads called for WebRtcSinkBintr '" 
            << static_cast<WebRtcSinkBintr*>(pWebRtcSink)->GetName() << "'");
    }

    static void on_new_transceiver_cb(G_GNUC_UNUSED GstElement* pWebrtcbin, 
        GstWebRTCRTPTransceiver* pTransceiver, gpointer pWebRtcSink)
    {
        LOG_INFO("on-new-transceiver called for WebRtcSinkBintr '" 
            << static_cast<WebRtcSinkBintr*>(pWebRtcSink)->GetName() << "'");
    }

    static void on_negotiation_needed_cb(GstElement* pWebrtcbin, gpointer pWebRtcSink)
    {
        static_cast<WebRtcSinkBintr*>(pWebRtcSink)->OnNegotiationNeeded();
    }

    static void on_offer_created_cb(GstPromise* pPromise, gpointer pWebRtcSink)
    {
        static_cast<WebRtcSinkBintr*>(pWebRtcSink)->OnOfferCreated(pPromise);
    }

    static void on_local_desc_set_cb(GstPromise* pPromise, gpointer pWebRtcSink)
    {
        static_cast<WebRtcSinkBintr*>(pWebRtcSink)->OnLocalDescSet(pPromise);
    }

    static void on_remote_desc_set_cb(GstPromise* pPromise, gpointer pWebRtcSink)
    {
        static_cast<WebRtcSinkBintr*>(pWebRtcSink)->OnRemoteDescSet(pPromise);
    }

    static void on_ice_candidate_cb(G_GNUC_UNUSED GstElement* pWebrtcbin, 
        guint mlineIndex, gchar * candidateStr, gpointer pWebRtcSink)
    {
        static_cast<WebRtcSinkBintr*>(pWebRtcSink)->
            OnIceCandidate(mlineIndex, candidateStr);
    }

    static void on_data_channel_cb(G_GNUC_UNUSED GstElement* pWebrtcbin, 
        GObject* pDataChannel, gpointer pWebRtcSink)
    {
        static_cast<WebRtcSinkBintr*>(pWebRtcSink)->ConnectDataChannelSignals(pDataChannel);
    }

    static void data_channel_on_error_cb(GObject* pDataChannel, gpointer pWebRtcSink)
    {
        LOG_ERROR("on-data-channel-errror called for WebRtcSinkBintr '" 
            << static_cast<WebRtcSinkBintr*>(pWebRtcSink)->GetName() << "'");

        // TODO: define and implement proper behavior
    }

    static void data_channel_on_open_cb(GObject* pDataChannel, gpointer pWebRtcSink)
    {
        static_cast<WebRtcSinkBintr*>(pWebRtcSink)->DataChannelOnOpen(pDataChannel);
    }

    static void data_channel_on_close_cb(GObject* pDataChannel, gpointer pWebRtcSink)
    {
        static_cast<WebRtcSinkBintr*>(pWebRtcSink)->DataChannelOnClose(pDataChannel);
    }

    static void data_channel_on_message_string_cb(GObject* pDataChannel, 
        gchar* messageStr, gpointer pWebRtcSink)
    {
        LOG_INFO("data-channel-on-message-string called for WebRtcSinkBintr '" 
            << static_cast<WebRtcSinkBintr*>(pWebRtcSink)->GetName() << "'");
        LOG_INFO("recieved message '" << messageStr << "'");
    }

    static int complete_on_closed_cb(gpointer pWebRtcSink)
    {
        return static_cast<WebRtcSinkBintr*>(pWebRtcSink)->CompleteOnClosed();
    }

} // DSL
