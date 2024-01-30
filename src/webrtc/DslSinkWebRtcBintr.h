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
         * @brief returns the current sync enabled property for the SinkBintr.
         * @return true if the sync property is enabled, false othewise.
         */
        gboolean GetSyncEnabled()
        {
            LOG_FUNC();
            return m_pFakeSinkBintr->GetSyncEnabled();  
        };
        
        /**
         * @brief sets the sync enabled property for the SinkBintr.
         * @param[in] enabled new sync enabled property value.
         */
        bool SetSyncEnabled(bool enabled)
        {
            LOG_FUNC();
            return m_pFakeSinkBintr->SetSyncEnabled(enabled);  
        };
        
        /**
         * @brief returns the current async enabled property value for the SinkBintr.
         * @return true if the async property is enabled, false othewise.
         */
        gboolean GetAsyncEnabled()
        {
            LOG_FUNC();
            return m_pFakeSinkBintr->GetAsyncEnabled();  
        };
        
        /**
         * @brief sets the async enabled property for the SinkBintr.
         * @param[in] enabled new async property value.
         */
        bool SetAsyncEnabled(gboolean enabled)
        {
            LOG_FUNC();
            return m_pFakeSinkBintr->SetAsyncEnabled(enabled);  
        };
        
        /**
         * @brief returns the current max-lateness property value for the SinkBintr.
         * @return current max-lateness (default = -1 unlimited).
         */
        gint64 GetMaxLateness()        
        {
            LOG_FUNC();
            return m_pFakeSinkBintr->GetMaxLateness();  
        };
        
        /**
         * @brief sets the max-lateness property for the SinkBintr.
         * @param[in] maxLateness new max-lateness proprty value.
         */
        bool SetMaxLateness(gint64 maxLateness)        
        {
            LOG_FUNC();
            return m_pFakeSinkBintr->SetMaxLateness(maxLateness);  
        };

        /**
         * @brief returns the current qos enabled property value for the SinkBintr.
         * @return true if the qos property is enabled, false othewise.
         */
        gboolean GetQosEnabled()        
        {
            LOG_FUNC();
            return m_pFakeSinkBintr->GetQosEnabled();  
        };
        
        /**
         * @brief sets the qos enabled property for the SinkBintr.
         * @param[in] enabled new qos enabled property value.
         */
        bool SetQosEnabled(gboolean enabled)        
        {
            LOG_FUNC();
            return m_pFakeSinkBintr->SetQosEnabled(enabled);  
        };

        bool CloseConnection();

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
         * @brief Called when the Websocket is closed
         * @param[in] pConnection unique connection that closed 
         */
        void OnClosed(SoupWebsocketConnection* pConnection);

        int CompleteOnClosed();

        /**
         * @brief Handles an incoming Websocket message
         * @param[in] pConnection pointer to the Websocket connection object.
         * @param[in] dataType type of the message received.
         * @param[in] message incoming message to handle.
         */
        void OnMessage(SoupWebsocketConnection* pConnection, 
            SoupWebsocketDataType dataType, GBytes* message);

        /**
         * @brief Handles the on-negotiation-needed by emitting a create-offer signal
         */
        void OnNegotiationNeeded();

        /**
         * @brief Handles the on-offer-created callback by emitting a "set-local-description"
         * signal and replying to he remote client
         * @param[in] promise 
         */
        void OnOfferCreated(GstPromise* pPromise);

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
        void OnLocalDescSet(GstPromise* pPromise);

        /**
         * @brief Handles the on-remote-desc-set callback by logging the reply
         * as INFO
         * @param[in] pPromise the promise from the initial offer.
         */
        void OnRemoteDescSet(GstPromise* pPromise);

        /**
         * @brief Common function to connect the Data Channel Signals
         * of channel setup
         */
        void ConnectDataChannelSignals(GObject* pDataChannel);

        /**
         * @brief Handles the data-channel-on-open signal by emitting a
         * signals to inform the remote client that the channel is ready. 
         */
        void DataChannelOnOpen(GObject* pDataChannel);

        /**
         * @brief Handles the data-channel-on-close signal by disconnecting
         * all data channel signal handlers. 
         */
        void DataChannelOnClose(GObject* pDataChannel);

    private:

        /** 
         * @brief WebRTC data channel for this WebRtcSinkBintr, 
         * NULL until channel has been setup.
         */
        GstWebRTCDataChannel* m_pDataChannel;

        /**
         * @brief gnome timer Id for the RTSP reconnection manager
         */
        uint m_completeClosedTimerId;


        /**
         * @brief Private function to iterate through the map of client listners
         * notifying each of the new/current state on change of state. 
         */
        void notifyClientListeners();

        /**
         * @brief Helper function to convert a json object to string
         * @return json string.
         */
        gchar* getStrFromJsonObj(JsonObject * object);

        /** 
         * @brief Handler Id for the RTP data chanel on-error signal handler,
         * 0 when not set
         */
        gulong m_dataChannelOnErrorSignalHandlerId;

        /** 
         * @brief Handler Id for the RTP data chanel on-open signal handler,
         * 0 when not set
         */
        gulong m_dataChannelOnOpenSignalHandlerId;

        /** 
         * @brief Handler Id for the RTP data chanel on-close signal handler,
         * 0 when not set
         */
        gulong m_dataChannelOnCloseSignalHandlerId;

        /** 
         * @brief Handler Id for the RTP data chanel on-message signal handler,
         * 0 when not set
         */
        gulong m_dataChannelOnMessageSignalHandlerId;

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
        std::map<dsl_sink_webrtc_client_listener_cb, void*> m_clientListeners;
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

    /**
     * @brief Callback function called on new WebRTC RTP Transciever.
     * @param[in] pWebrtcbin pointer to the webrtcbin element connected to the Transciever.
     * @param[in] pTransceiver pointer to the new RTP Transciever.
     * @param[in] pWebRtcSink pointer to the WebRtcSinkBintr that owns the webrtcbin.
     */
    static void on_new_transceiver_cb(G_GNUC_UNUSED GstElement* pWebrtcbin, 
        GstWebRTCRTPTransceiver* pTransceiver, gpointer pWebRtcSink);

    /**
     * @brief Callback function called on negotion needed.
     * @param[in] pWebrtcbin pointer to the webrtcbin element connected to the Transciever.
     * @param[in] pWebRtcSink pointer to the WebRtcSinkBintr that owns the webrtcbin.
     */
    static void on_negotiation_needed_cb(GstElement* pWebrtcbin, gpointer pWebRtcSink);

    /**
     * @brief Callback function called on offer created 
     * @param[in] pPromise pointer to the promise ??
     * @param[in] pWebRtcSink pointer to the WebRtcSinkBintr that owns the webrtcbin.
     */
    static void on_offer_created_cb(GstPromise* pPromise, gpointer pWebRtcSink);

    /**
     * @brief Callback function called on local description set
     * @param[in] pPromise pointer to the promise ??.
     * @param[in] pWebRtcSink pointer to the WebRtcSinkBintr that owns the webrtcbin.
     */
    static void on_local_desc_set_cb(GstPromise* pPromise, gpointer pWebRtcSink);

    /**
     * @brief Callback function called on remote description set
     * @param[in] pPromise pointer to the promise with the reply to the offer created??
     * @param[in] pWebRtcSink pointer to the WebRtcSinkBintr that owns the webrtcbin.
     */
    static void on_remote_desc_set_cb(GstPromise* pPromise, gpointer pWebRtcSink);

    /**
     * @brief Callback function called on ICE candidate recieved
     * @param[in] pWebrtcbin pointer to the webrtcbin element connected to the RTP Transciever.
     * @param[in] mlineIndex line index for the candidate string.
     * @param[in] candidateStr the ICE candidate info string.
     * @param[in] pWebRtcSink pointer to the WebRtcSinkBintr that owns the webrtcbin.
     */
    static void on_ice_candidate_cb(G_GNUC_UNUSED GstElement* pWebrtcbin, 
        guint mlineIndex, gchar * candidateStr, gpointer pWebRtcSink);

    /**
     * @brief Callback function called on new data channel.
     * @param[in] pWebrtcbin pointer to the webrtcbin element connected to the data channel.
     * @param[in] pDataChannel pointer to the data channel created.
     * @param[in] pWebRtcSink pointer to the WebRtcSinkBintr that owns the webrtcbin.
     */
    static void on_data_channel_cb(G_GNUC_UNUSED GstElement* pWebrtcbin, 
        GObject* pDataChannel, gpointer pWebRtcSink);

    /**
     * @brief Callback function called on data channel error.
     * @param[in] pDataChannel pointer to the data channel in error.
     * @param[in] pWebRtcSink pointer to the WebRtcSinkBintr that owns the data channel.
     */
    static void data_channel_on_error_cb(GObject* pDataChannel, gpointer pWebRtcSink);

    /**
     * @brief Callback function called on data channel opened.
     * @param[in] pDataChannel pointer to the data channel that closed.
     * @param[in] pWebRtcSink pointer to the WebRtcSinkBintr that owns the data channel.
     */
    static void data_channel_on_open_cb(GObject* pDataChannel, gpointer pWebRtcSink);

    /**
     * @brief Callback function called on data channel closed.
     * @param[in] pDataChannel pointer to the data channel that closed.
     * @param[in] pWebRtcSink pointer to the WebRtcSinkBintr that owns the data channel.
     */
    static void data_channel_on_close_cb(GObject* pDataChannel, gpointer pWebRtcSink);

    /**
     * @brief Callback function called on new incomming message string.
     * @param[in] pDataChannel pointer to the data channel the message was received on.
     * @param[in] messageStr the recieved message string.
     * @param[in] pWebRtcSink pointer to the WebRtcSinkBintr that owns the data channel.
     */
    static void data_channel_on_message_string_cb(GObject* dataChannel, 
        gchar* messageStr, gpointer pWebRtcSink);

    static int complete_on_closed_cb(gpointer pWebRtcSink);

}
#endif //_DSL_SINK_BINTR_H
