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

#ifndef _DSL_SOUP_SERVER_H
#define _DSL_SOUP_SERVER_H

#include "Dsl.h"
#include <gst/sdp/sdp.h>
#include <libsoup/soup.h>
#include <json-glib/json-glib.h>

#define GST_USE_UNSTABLE_API
#include <gst/webrtc/webrtc.h>

#include "DslSinkBintr.h"

namespace DSL
{
    struct ClientReceiver
    {
        /**
         * @brief Ctor for the ClientReceiver
         */
        ClientReceiver(gpointer pBin)
            : pBin(pBin)
            , pConnection(NULL)
            , pOffer(NULL)
            , pSendChannel(NULL)
        {};

        /** 
         * @brief webrtcbin owned by this client
         */
        gpointer pBin;

        /** 
         * @brief Client's unique Websocket connection, NULL until connection established.
         */
        SoupWebsocketConnection* pConnection;

        /** 
         * @brief .
         */
        GstWebRTCSessionDescription* pOffer;

        /** 
         * @brief Client's send data channel for the Websocket connection, 
         * NULL until channel has been setup.
         */
        GstWebRTCDataChannel* pSendChannel;
    };

    class SoupServerMgr
    {
    public: 

        /** 
         * @brief Returns a pointer to this singleton Manager
         * @return instance pointer to Services
         */
        static SoupServerMgr* GetMgr();

        /**
         * @brief Ctor for this singleton SoupServerMgr class
         */
        SoupServerMgr();

        /**
         * @brief Dtor for this singleton SoupServerMgr class
         */
        ~SoupServerMgr();

        /**
         * @brief Adds a client webrtcbin to this server
         * @param [in] pClientBin unique webrtcbin to add
         * @return true on successful add, false otherwise
         */
        bool AddClientBin(gpointer pClientBin);

        /**
         * @brief Removes a client webrtcbin from this server
         * @param [in] pClientBin unique webrtcbin to remove
         * @return true on successful remove, false otherwise
         */
        bool RemoveClientBin(gpointer pClientBin);

        /**
         * @brief Checks whether a client webrtcbin has been previously 
         * added to this Soup Server Manager
         * @param [in] pClientBin unique webrtcbin to check for
         * @return true if found, false otherwise
         */
        bool HasClientBin(gpointer pClientBin);

        /**
         * @brief Gets the client webrtcbin for a specifed connection
         * @return the client webrtcbin if connection found, NULL ohterwise.
         */
        gpointer GetClientBin(SoupWebsocketConnection* pConnection);

        /**
         * @brief Handles a new Websocket Connection
         * @param[in] pConnection unique connection to open 
         */
        void HandleOpen(SoupWebsocketConnection* pConnection);

        /**
         * @brief Handles the closing of a Websocket Connection
         * @param[in] pConnection unique connection to open 
         */
        void HandleClose(SoupWebsocketConnection* pConnection);

        /**
         * @brief Handles a message from a Websocket Connection
         * @param[in] pConnection unique connection for this message
         */
        void HandleMessage(SoupWebsocketConnection* pConnection, 
            SoupWebsocketDataType dataType, GBytes* message);

        /**
         * @brief Handles the on-reset-desc-set callback by
         * @param[in] promise 
         */
        void OnRemoteDescSet(GstPromise* promise);

    private:

        /**
         * @brief instance pointer for this singleton class
         */
        static SoupServerMgr* m_pInstance;

        /**
         * @brief Soup Server instance used by this singlton.
         */
        SoupServer* m_pSoupServer;

        /**
         * @brief JSON object parser used by this singlton.
         */
        JsonParser* m_pJsonParser;
    
        /**
         * @brief mutex to protect mutual access to queue data
         */
        GMutex m_serverMutex;

        /**
         * @brief container of all client receivers mapped by their unique webrtcbin
         * connection, promise, and send-channel data.
         */
        std::map<gpointer, std::shared_ptr<ClientReceiver>> m_clientReceivers;
    };

    static void on_soup_websocket_opened_cb(G_GNUC_UNUSED SoupServer* pServer, 
        SoupWebsocketConnection* pConnection, G_GNUC_UNUSED const char *path,
        G_GNUC_UNUSED SoupClientContext* clientContext, gpointer pSoupServerMgr);

    static void on_soup_websocket_closed_cb(SoupWebsocketConnection * pConnection, 
        gpointer pSoupServerMgr);

    static void on_soup_websocket_message_cb(SoupWebsocketConnection* pConnection, 
        SoupWebsocketDataType dataType, GBytes* message, gpointer pSoupServerMgr);

    static void on_remote_desc_set_cb(GstPromise * promise, gpointer pSoupServerMgr);


} // DSL
#endif // _DSL_SOUP_SERVER_H