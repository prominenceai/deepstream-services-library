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
        const char* turnServer, uint codec, uint bitrate, uint interval)
        : EncodeSinkBintr(name, codec, bitrate, interval)
        , SignalingTransceiver()
        , m_stunServer(stunServer)
        , m_turnServer(turnServer)
    {
        LOG_FUNC();

        std::string fakeSinkName = GetName() + "-fake-sink";
        m_pFakeSinkBintr = DSL_FAKE_SINK_NEW(fakeSinkName.c_str());

        switch (codec)
        {
        case DSL_CODEC_H264 :
            m_pPayloader = DSL_ELEMENT_NEW("rtph264pay", "webrtc-sink-bin-h264-payloader");
            break;
        case DSL_CODEC_H265 :
            m_pPayloader = DSL_ELEMENT_NEW("rtph265pay", "webrtc-sink-bin-h265-payloader");
            break;
        default:
            LOG_ERROR("Invalid codec = '" << codec << "' for new WebRtcSinkBintr '" << name << "'");
            throw;
        }
        m_pWebRtcCapsFilter = DSL_ELEMENT_NEW(NVDS_ELEM_CAPS_FILTER, "webrtc-sink-bin-caps-filter");
        
        GstCaps* pCaps = gst_caps_from_string("application/x-rtp,media=video,encoding-name=H264,payload=96");
        m_pWebRtcCapsFilter->SetAttribute("caps", pCaps);
        gst_caps_unref(pCaps);

        m_pWebRtcBin = DSL_ELEMENT_NEW("webrtcbin", "sink-bin-webrtc");

        // Set the STUN or TURN server 
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

        SoupServerMgr::GetMgr()->AddSignalingTransceiver(this);
        
        AddChild(m_pPayloader);
        AddChild(m_pWebRtcCapsFilter);
        AddChild(m_pWebRtcBin);
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
            return false;
        }
        if (!m_pQueue->LinkToSink(m_pTransform) or
            !m_pTransform->LinkToSink(m_pCapsFilter) or
            !m_pCapsFilter->LinkToSink(m_pEncoder) or
            !m_pEncoder->LinkToSink(m_pParser) or
            !m_pParser->LinkToSink(m_pPayloader) or
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
        m_pWebRtcCapsFilter->UnlinkFromSink();
        m_pPayloader->UnlinkFromSink();
        m_pParser->UnlinkFromSink();
        m_pEncoder->UnlinkFromSink();
        m_pCapsFilter->UnlinkFromSink();
        m_pTransform->UnlinkFromSink();
        m_pQueue->UnlinkFromSink();
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
        GstState state;
        pParentBranchBintr->GetState(state, 0);
        if (state == GST_STATE_PLAYING)
        {
            // add the this WebRtcSinkBintr now
            if (!pParentBranchBintr->AddSinkBintr(
                    std::dynamic_pointer_cast<SinkBintr>(shared_from_this())))
            {
                LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                    << "' failed to add its FakeSinkBintr to the parent branch");
                return false;
            }
        }
        // Setup the current the Parent Pipeline/Branch pointer
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

        if (state == GST_STATE_PLAYING)
        {
            // add the this WebRtcSinkBintr now
            if (!pParentBranchBintr->AddSinkBintr(
                    std::dynamic_pointer_cast<SinkBintr>(shared_from_this())))
            {
                LOG_ERROR("WebRtcSinkBintr '" << GetName() 
                    << "' failed to add its self to the parent branch");
                return;
            }
        }
        else
        {
            LOG_ERROR("The WebRtcSinkBintr '" << GetName() 
                << "' is not a child of any parent branch or pipeline");
            return;
        }

        // If we have registered client listerns for the WebRtSinkBintr
        if (m_clientListeners.size())
        {
            // setup the connection information data for the client listener(s)
            dsl_webrtc_connection_data data{0};

            // single field for now
            data.current_state = m_connectionState;

            // iterate through the map of Termination event listeners calling each
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

    bool WebRtcSinkBintr::SetSyncSettings(bool sync, bool async)
    {
        LOG_FUNC();
        
        if (IsLinked())
        {
            LOG_ERROR("Unable to set Sync/Async Settings for WebRtcSinkBintr '" 
                << GetName() << "' as it's currently linked");
            return false;
        }
        m_sync = sync;
        m_async = async;

        // Nothing to set for webrtcbin??
        return true;
    }

    void WebRtcSinkBintr::OnNegotiationNeeded()
    {
        LOG_FUNC();

        LOG_INFO("on-negotiation-needed called for WebRtcSinkBintr '" 
            << GetName() << "'");

        GstPromise* promise = gst_promise_new_with_change_func(
            on_offer_created_cb, (gpointer)this, NULL);
        g_signal_emit_by_name(m_pWebRtcBin->GetGstObject(), 
            "create-offer", NULL, promise);
    }

    void WebRtcSinkBintr::OnOfferCreated(GstPromise* promise)
    {
        LOG_FUNC();

        LOG_INFO("on-offer-created called for WebRtcSinkBintr '" 
            << GetName() << "'");

        GstStructure const* reply = gst_promise_get_reply(promise);
        GstWebRTCSessionDescription *offer = NULL;
        gst_structure_get(reply, "offer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, NULL);
        gst_promise_unref(promise);

        GstPromise* localDescPromise = gst_promise_new_with_change_func(
                on_local_desc_set_cb, (gpointer)this, NULL);
        g_signal_emit_by_name(m_pWebRtcBin->GetGstObject(), 
            "set-local-description", offer, localDescPromise);

        gchar* sdpStr = gst_sdp_message_as_text(offer->sdp);

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

        gst_webrtc_session_description_free(offer);  
    }

    void WebRtcSinkBintr::OnIceCandidate(guint mLineIndex, gchar* candidate)
    
    {
        LOG_FUNC();

        LOG_INFO("on-ice-candidate '" << candidate << "' received for WebRtcSinkBintr '" 
            << GetName() << "'");

        JsonObject* ice_json = json_object_new();
        json_object_set_string_member(ice_json, "type", "ice");

        JsonObject* ice_data_json = json_object_new();
        json_object_set_int_member(ice_data_json, "sdpMLineIndex", mLineIndex);
        json_object_set_string_member(ice_data_json, "candidate", candidate);
        json_object_set_object_member(ice_json, "data", ice_data_json);

        gchar* jsonStr = getStrFromJsonObj(ice_json);
        json_object_unref(ice_json);

        soup_websocket_connection_send_text(m_pConnection, jsonStr);
        g_free(jsonStr);
    }

    void WebRtcSinkBintr::OnLocalDescSet(GstPromise* promise)
    {
        LOG_FUNC();

        LOG_INFO("on-local-desc-set called for WebRtcSinkBintr '" 
            << GetName() << "'");

        GstStructure const *reply = gst_promise_get_reply(promise);
        if (reply != NULL)
        {
            gchar* replyStr = gst_structure_to_string(reply);
            LOG_INFO("Reply for on-local-desc-set is '" << replyStr
                << "' for WebRtcSinkBintr '" << GetName());
            g_free(replyStr);
        }
        gst_promise_unref(promise);  
    }

    void WebRtcSinkBintr::OnRemoteDescSet(GstPromise* promise)
    {
        LOG_FUNC();

        LOG_INFO("on-remote-desc-set called for WebRtcSinkBintr '" 
            << GetName() << "'");

        GstStructure const *reply = gst_promise_get_reply(promise);
        if (reply != NULL)
        {
            gchar* replyStr = gst_structure_to_string(reply);
            LOG_INFO("Reply for on-remote-desc-set is '" << replyStr
                << "' for WebRtcSinkBintr '" << GetName());
            g_free(replyStr);
        }
        gst_promise_unref(promise);  
    }

    void WebRtcSinkBintr::HandleMessage(SoupWebsocketConnection* pConnection, 
        SoupWebsocketDataType dataType, GBytes* message)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_receiverMutex);

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
            LOG_ERROR("Soup Server Manager received unknown data type");
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

            GstPromise* promise = gst_promise_new_with_change_func(on_remote_desc_set_cb, 
                m_pWebRtcBin->GetGstObject(), NULL);

            g_signal_emit_by_name(G_OBJECT(m_pWebRtcBin->GetGObject()), 
                "set-remote-description", answer, promise);    
            gst_webrtc_session_description_free(answer);

            // if(receiver_entry->send_channel == NULL)
            // {
            //     g_signal_emit_by_name (receiver_entry->webrtcbin, "create-data-channel", "channel", NULL, &receiver_entry->send_channel);
            //     if (receiver_entry->send_channel) 
            //     {
            //         gst_print ("Created data channel\n");
            //         connect_data_channel_signals((GObject*)receiver_entry->send_channel);
            //     }
            //     else 
            //     {
            //         gst_print ("Could not create data channel, is usrsctp available?\n");
            //     }
            // }
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

    static void on_pad_added_cb(GstElement * webrtcbin, GstPad* pad, gpointer pWebRtcSink)
    {
        LOG_INFO("on-pad-added called for WebRtcSinkBintr '"
            << static_cast<WebRtcSinkBintr*>(pWebRtcSink)->GetName() << "'");
    }

    static void on_pad_removed_cb(GstElement * webrtcbin, GstPad* pad, gpointer pWebRtcSink)
    {
        LOG_INFO("on-pad-removed called for WebRtcSinkBintr '" 
            << static_cast<WebRtcSinkBintr*>(pWebRtcSink)->GetName() << "'");
    }
 
    static void on_no_more_pads_cb(GstElement * webrtcbin, gpointer pWebRtcSink)
    {
        LOG_WARN("on-no-more-pads called for WebRtcSinkBintr '" 
            << static_cast<WebRtcSinkBintr*>(pWebRtcSink)->GetName() << "'");
    }

    static void on_negotiation_needed_cb(GstElement* webrtcbin, gpointer pWebRtcSink)
    {
        static_cast<WebRtcSinkBintr*>(pWebRtcSink)->OnNegotiationNeeded();
    }

    static void on_offer_created_cb(GstPromise* promise, gpointer pWebRtcSink)
    {
        static_cast<WebRtcSinkBintr*>(pWebRtcSink)->OnOfferCreated(promise);
    }

    static void on_local_desc_set_cb(GstPromise* promise, gpointer pWebRtcSink)
    {
        static_cast<WebRtcSinkBintr*>(pWebRtcSink)->OnLocalDescSet(promise);
    }

    static void on_remote_desc_set_cb(GstPromise* promise, gpointer pWebRtcSink)
    {
        static_cast<WebRtcSinkBintr*>(pWebRtcSink)->OnRemoteDescSet(promise);
    }

    static void on_ice_candidate_cb(G_GNUC_UNUSED GstElement * webrtcbin, 
        guint mline_index, gchar * candidate, gpointer pWebRtcSink)
    {
        static_cast<WebRtcSinkBintr*>(pWebRtcSink)->
            OnIceCandidate(mline_index, candidate);
    }

    static void on_new_transceiver_cb(GstElement* webrtcbin, 
        GstWebRTCRTPTransceiver* transceiver, gpointer pWebRtcSink)
    {
        LOG_INFO("on-new-transceiver called for WebRtcSinkBintr '" 
            << static_cast<WebRtcSinkBintr*>(pWebRtcSink)->GetName() << "'");
    }

    static void on_data_channel_cb(GstElement* webrtc, GObject* data_channel, gpointer pWebRtcSink)
    {
        gst_print("webrtcbin::on_data_channel\n");  

        g_signal_connect(data_channel, "on-error", G_CALLBACK(data_channel_on_error_cb), pWebRtcSink);
        g_signal_connect(data_channel, "on-open", G_CALLBACK(data_channel_on_open_cb), pWebRtcSink);
        g_signal_connect(data_channel, "on-close", G_CALLBACK(data_channel_on_close_cb), pWebRtcSink);
        g_signal_connect(data_channel, "on-message-string", G_CALLBACK(data_channel_on_message_string_cb), pWebRtcSink);
    }

    static void data_channel_on_error_cb(GObject* dc, gpointer pWebRtcSink)
    {
        gst_print("webrtcbin::data_channel_on_error\n");  
        //cleanup_and_quit_loop("Data channel error", 0);
    }

    static void data_channel_on_open_cb(GObject* dc, gpointer pWebRtcSink)
    {
        GstWebRTCDataChannel* dataChannel = (GstWebRTCDataChannel*)dc;
        gst_print("webrtcbin::data_channel_on_open: %s (%d)\n", dataChannel->label, dataChannel->ready_state);  

        GBytes *bytes = g_bytes_new("data", strlen ("data"));
        g_signal_emit_by_name(dc, "send-string", "Equature AI Ready");
        g_signal_emit_by_name(dc, "send-data", bytes);
        g_bytes_unref(bytes);
    }

    static void data_channel_on_close_cb(GObject* dc, gpointer user_data)
    {
        gst_print("webrtcbin::data_channel_on_close\n");  

        //cleanup_and_quit_loop ("Data channel closed", 0);
    }

    static void data_channel_on_message_string_cb(GObject* dc, gchar* str, gpointer user_data)
    {
        gst_print("webrtcbin::data_channel_on_message_string\n");  

        gst_print ("Received data channel message: %s\n", str);
    }


} // DSL