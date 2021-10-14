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

namespace DSL
{

    #define DSL_WEBRTC_SINK_PTR std::shared_ptr<WebRtcSinkBintr>
    #define DSL_WEBRTC_SINK_NEW(name) \
        std::shared_ptr<WebRtcSinkBintr>(new WebRtcSinkBintr(name))

    class WebRtcSinkBintr : public SinkBintr
    {
    public: 
    
        WebRtcSinkBintr(const char* name);

        ~WebRtcSinkBintr();
  
        /**
         * @brief Links all Child Elementrs owned by this Bintr
         * @return true if all links were succesful, false otherwise
         */
        bool LinkAll();
        
        /**
         * @brief Unlinks all Child Elemntrs owned by this Bintr
         * Calling UnlinkAll when in an unlinked state has no effect.
         */
        void UnlinkAll();

        /**
         * @brief sets the current sync and async settings for the SinkBintr
         * @param[in] sync current sync setting, true if set, false otherwise.
         * @param[in] async current async setting, true if set, false otherwise.
         * @return true is successful, false otherwise. 
         */
        bool SetSyncSettings(bool sync, bool async);

        /**
         * @brief Handles the on-negotiation-needed by emitting a create-offer signal
         */
        void OnNegotiationNeeded();

        /**
         * @brief Handles the on-offer-created callback by
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
         * @brief Handles the on-local-desc-set callback by
         * @param[in] promise 
         */
        void OnLocalDescSet(GstPromise* promise);

    private:

        gchar* getStrFromJsonObj(JsonObject * object);

        SoupWebsocketConnection* m_pConnection;

        GstWebRTCDataChannel* m_pSendChannel;   
        
        /**
         * @brief WebRtc bin element for this WebRtcSinkBintr.
         */
        DSL_ELEMENT_PTR m_pWebRtcBin;
    };

    static void on_pad_added_cb(GstElement * webrtcbin, GstPad* pad, gpointer pWebRtcSink);

    static void on_pad_removed_cb(GstElement * webrtcbin, GstPad* pad, gpointer pWebRtcSink);

    static void on_no_more_pads_cb(GstElement * webrtcbin, gpointer user_data);

    static void on_negotiation_needed_cb(GstElement * webrtcbin, gpointer pWebRtcSink);

    static void on_offer_created_cb(GstPromise* promise, gpointer pWebRtcSink);

    static void on_local_desc_set_cb(GstPromise* promise, gpointer pWebRtcSink);

    static void on_ice_candidate_cb(G_GNUC_UNUSED GstElement * webrtcbin, 
        guint mline_index, gchar * candidate, gpointer pWebRtcSink);

    static void on_new_transceiver_cb(GstElement * webrtcbin, 
        GstWebRTCRTPTransceiver* transceiver, gpointer pWebRtcSink);

}
#endif //_DSL_SINK_BINTR_H
