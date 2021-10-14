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
#include "DslSinkWebRtcBintr.h"
#include "DslSoupServerMgr.h"

namespace DSL
{
    WebRtcSinkBintr::WebRtcSinkBintr(const char* name)
        : SinkBintr(name, true, false)
    {
        LOG_FUNC();
        
        m_pWebRtcBin = DSL_ELEMENT_NEW("webrtcbin", "sink-bin-webrtc");

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
        // g_signal_connect(m_pWebRtcBin->GetGstObject(), "on-data-channel",
        //     G_CALLBACK(on_data_channel), (gpointer)this);

        SoupServerMgr::GetMgr()->AddClientBin(m_pWebRtcBin->GetGstObject());
        
        AddChild(m_pWebRtcBin);
    }
    
    WebRtcSinkBintr::~WebRtcSinkBintr()
    {
        LOG_FUNC();
    
        SoupServerMgr::GetMgr()->RemoveClientBin(m_pWebRtcBin->GetGstObject());
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
        if (!m_pQueue->LinkToSink(m_pWebRtcBin))
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
        m_pQueue->UnlinkFromSink();
        m_isLinked = false;
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

        GstPromise* promise = gst_promise_new_with_change_func(
            on_offer_created_cb, (gpointer)this, NULL);
        g_signal_emit_by_name(m_pWebRtcBin->GetGstObject(), 
            "create-offer", NULL, promise);
    }

    void WebRtcSinkBintr::OnOfferCreated(GstPromise* promise)
    {
        LOG_FUNC();

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

    static void on_ice_candidate_cb(G_GNUC_UNUSED GstElement * webrtcbin, 
        guint mline_index, gchar * candidate, gpointer pWebRtcSink)
    {
        static_cast<WebRtcSinkBintr*>(pWebRtcSink)->
            OnIceCandidate(mline_index, candidate);
    }

    static void on_new_transceiver_cb(GstElement * webrtcbin, 
        GstWebRTCRTPTransceiver* transceiver, gpointer pWebRtcSink)
    {
        LOG_WARN("on-new-transceiver called for WebRtcSinkBintr '" 
            << static_cast<WebRtcSinkBintr*>(pWebRtcSink)->GetName() << "'");
    }

} // DSL