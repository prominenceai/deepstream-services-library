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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef _DSL_SINK_WEBRTC_BINTR_H
#define _DSL_SINK_WEBRTC_BINTR_H

#include "Dsl.h"
#include <gst/sdp/sdp.h>
#include <libsoup/soup.h>
#include <json-glib/json-glib.h>

#define GST_USE_UNSTABLE_API
#include <gst/webrtc/webrtc.h>

#include "DslSinkBintr.h"
#include "DslSoupServerMgr.h"

namespace DSL
{

    #define DSL_WEBRTC_SINK_PTR std::shared_ptr<WebRtcSinkBintr>
    #define DSL_WEBRTC_SINK_NEW(name, stunServer, turnServer, \
        codec, bitrate, interval) \
        std::shared_ptr<WebRtcSinkBintr>(new WebRtcSinkBintr(name, \
            stunServer, turnServer, codec, bitrate, interval))

    /**
     * @class WebRtcSinkBintr 
     * @file DslWebRtcSinkBintr.h
     * @brief Implements a WebRTC Sink Bin Container Class (Bintr)
     */
    class WebRtcSinkBintr : public EncodeSinkBintr, public SignalingTransceiver
    {
    public: 
    
        /**
         * @brief Ctor for the WebRtcSinkBintr class
         */
        WebRtcSinkBintr(const char* name, const char* stunServer, 
            const char* turnServer, uint container, uint bitRate, uint interval);

        /**
         * @brief Dtor for the WebRtcSinkBintr class
         */
        ~WebRtcSinkBintr();
  
        /**
         * @brief Links all Child Elementrs owned by this WebRtcSinkBintr
         * @return true if all links were succesful, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elemntrs owned by this WebRtcSinkBintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();

        /**
         * @brief adds this WebRtcSinkBintr to a parent Branch/Pipeline bintr
         * @param[in] pParentBintr parent bintr to add this sink to
         * @return true on successful add, false otherwise
         */
        bool AddToParent(DSL_BASE_PTR pParentBintr);

        /**
         * @brief removes this WebRtcSinkBintr from a parent Branch/Pipeline bintr
         * @param[in] pParentBintr parent bintr to remove this sink from
         * @return true on successful remove, false otherwise
         */
        bool RemoveFromParent(DSL_BASE_PTR pParentBintr);

        /**
         * @brief sets the current sync and async settings for the WebRtcSinkBintr
         * @param[in] sync current sync setting, true if set, false otherwise.
         * @param[in] async current async setting, true if set, false otherwise.
         * @return true is successful, false otherwise. 
         */
        bool SetSyncSettings(bool sync, bool async);

        /**
         * @brief gets the current STUN and TURN server settings in use by 
         * the WebRtcSinkBintr
         * @param[out] stunServer current STUN Server setting in use
         * @param[out] turnServer current TURN Server setting in use
         */
        void GetServers(const char** stunServer, const char** turnServer);

        /**
         * @brief sets the current sync and async settings for the SinkBintr
         * @param[in] stunServer new STUN Server setting to use
         * @param[in] turnServer new TURN Server setting to use
         * @return true is successful set, false otherwise. 
         */
        bool SetServers(const char* stunServer, const char* turnServer);

        /**
         * @brief adds a callback to be notified on connection event
         * @param[in] listener pointer to the client's function to call on connection event
         * @param[in] clientData opaque pointer to client data passed into the listener function.
         * @return true on successful add, false otherwise
         */
        bool AddClientListener(dsl_sink_webrtc_client_listener_cb listener, 
            void* clientData);

        /**
         * @brief removes a previously added callback
         * @param[in] listener pointer to the client's function to remove
         * @return true on successful remove, false otherwise
         */
        bool RemoveClientListener(dsl_sink_webrtc_client_listener_cb listener);


        /**
         * @brief Sets the current connection for this SignalingTransceiver.
         * @param[in] pConnection pointer to Websocket Connection struction, 
         * set to NULL to reset.
         */
        void SetConnection(SoupWebsocketConnection* pConnection);

        /**
         * @brief Handles the on-negotiation-needed by emitting a create-offer signal
         */
        void OnNegotiationNeeded();

        /**
         * @brief Handles the on-offer-created callback by emitting a "set-local-description"
         * signal and replying to he remote client
         * @param[in] promise 
         */
        void OnOfferCreated(GstPromise* promise);

        /**
         * @brief Handles the on-ice-candidate callback by
         * @param[in] mLineIndex 
         * @param[in] candidate
         */
        void OnIceCandidate(guint mLineIndex, gchar* candidate);

        /**
         * @brief Handles the on-local-desc-set callback by logging the reply
         * as INFO
         * @param[in] promise the promise from the initial offer.
         */
        void OnLocalDescSet(GstPromise* promise);

        /**
         * @brief Handles the on-remote-desc-set callback by logging the reply
         * as INFO
         * @param[in] promise the promise from the initial offer.
         */
        void OnRemoteDescSet(GstPromise* promise);

        /**
         * @brief Handles an incoming Websocket message
         * @param[in] pConnection pointer to the Websocket connection object.
         * @param[in] dataType type of the message received.
         * @param[in] message incoming message to handle.
         */
        void HandleMessage(SoupWebsocketConnection* pConnection, 
            SoupWebsocketDataType dataType, GBytes* message);

    private:

        /**
         * @brief Client provided STUN server for this WebRtcSinkBintr 
         * of the form stun://hostname:port
         */
        std::string m_stunServer;

        /**
         * @brief [optional] Client provided TURN server for this WebRtcSinkBintr 
         * of the form turn(s)://username:password@host:port. 
         */
        std::string m_turnServer;

        gchar* getStrFromJsonObj(JsonObject * object);


        /**
         * @brief Pointer to the WebRtcSinks' channel for sending data, NULL until connected.
         */
        GstWebRTCDataChannel* m_pSendChannel;

        /**
         * @brief shared pointer to parent BranchBintr or PipelineBintr. nullptr if none.
         */
        DSL_BASE_PTR m_pParentBintr;

        /**
         * @brief FakeSinkBintr to add to the Pipline so that it can play prior
         * to adding this WebRtcSinkBintr
         */
        DSL_FAKE_SINK_PTR m_pFakeSinkBintr;

        /**
         * @brief Payloader element for this WebRtcSinkBintr.
         */
        DSL_ELEMENT_PTR m_pPayloader;

        /**
         * @brief webrtcbin sink-pad caps element for this WebRtcSinkBintr.
         */
        DSL_ELEMENT_PTR m_pWebRtcCapsFilter;

        /**
         * @brief webrtcbin element for this WebRtcSinkBintr.
         */
        DSL_ELEMENT_PTR m_pWebRtcBin;

        /**
         * @brief map of all currently registered client listeners
         * callback functions mapped with the user provided data
         */
        std::map<dsl_sink_webrtc_client_listener_cb, void*>m_clientListeners;

    };

    /**
     * @brief Callback function invoked when a pad is added to the webrtcbin.
     * @param[in] webrtcbin the webrtcbin instance the pad is added to
     * @param[in] pad the pad that was added to the webrtcbin
     * @param[in] pWebRtcSink pointer to the parent WebRtcSink that owns the webrtcbin
     */
    static void on_pad_added_cb(GstElement* webrtcbin, GstPad* pad, gpointer pWebRtcSink);

    /**
     * @brief Callback function invoked when a pad is removed from the webrtcbin.
     * @param[in] webrtcbin the webrtcbin instance the pad is removed from.
     * @param[in] pad the pad that was removed from the webrtcbin.
     * @param[in] pWebRtcSink pointer to the parent WebRtcSink that owns the webrtcbin.
     */
    static void on_pad_removed_cb(GstElement* webrtcbin, GstPad* pad, gpointer pWebRtcSink);

    /**
     * @brief Callback function invoked when a pad is removed from the webrtcbin.
     * @param[in] webrtcbin the webrtcbin instance the pad is removed from.
     * @param[in] pad the pad that was removed from the webrtcbin.
     * @param[in] pWebRtcSink pointer to the parent WebRtcSink that owns the webrtcbin.
     */
    static void on_no_more_pads_cb(GstElement* webrtcbin, gpointer user_data);

    static void on_negotiation_needed_cb(GstElement* webrtcbin, gpointer pWebRtcSink);

    static void on_offer_created_cb(GstPromise* promise, gpointer pWebRtcSink);

    static void on_local_desc_set_cb(GstPromise* promise, gpointer pWebRtcSink);

    static void on_remote_desc_set_cb(GstPromise* promise, gpointer pWebRtcSink);

    static void on_ice_candidate_cb(G_GNUC_UNUSED GstElement* webrtcbin, 
        guint mline_index, gchar* candidate, gpointer pWebRtcSink);

    static void on_new_transceiver_cb(GstElement* webrtcbin, 
        GstWebRTCRTPTransceiver* transceiver, gpointer pWebRtcSink);

    static void on_data_channel_cb(GstElement* webrtc, GObject* data_channel, gpointer pWebRtcSink);

    static void data_channel_on_open_cb(GObject* dc, gpointer pWebRtcSink);

    static void data_channel_on_error_cb(GObject* dc, gpointer pWebRtcSink);

    static void data_channel_on_close_cb(GObject* dc, gpointer pWebRtcSink);

    static void data_channel_on_message_string_cb(GObject* dc, gchar* str, gpointer pWebRtcSink);
}
#endif //_DSL_SINK_BINTR_H
