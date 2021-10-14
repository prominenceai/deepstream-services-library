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

#include "DslSoupServerMgr.h"

namespace DSL
{


    // Initialize the SoupServer single instance pointer
    SoupServerMgr* SoupServerMgr::m_pInstance = NULL;

    SoupServerMgr* SoupServerMgr::GetMgr()
    {
        // one time initialization of the single instance pointer
        if (!m_pInstance)
        {
            
            LOG_INFO("SoupServer Initialization");
            
            // Single instantiation for the lib's lifetime
            m_pInstance = new SoupServerMgr();
        }
        return m_pInstance;
    }
        
    SoupServerMgr::SoupServerMgr()
        : m_pSoupServer(NULL)
        , m_pJsonParser(NULL)

    {
        LOG_FUNC();

        m_pJsonParser = json_parser_new();
        if (!m_pJsonParser)
        {
            LOG_ERROR("Failed to create new JSON Parser");
            throw;
        }

        m_pSoupServer = soup_server_new(SOUP_SERVER_SERVER_HEADER, "webrtc", NULL);
        if (!m_pSoupServer)
        {
            LOG_ERROR("Failed to create new Soup Server");
            throw;
        }

        soup_server_add_websocket_handler(m_pSoupServer, "/ws", NULL, NULL, 
            on_soup_websocket_opened_cb, (gpointer)m_pInstance, NULL);
        soup_server_listen_all(m_pSoupServer, DSL_SOUP_HTTP_PORT, 
            (SoupServerListenOptions) 0, NULL);

        // Get and log the list of URIs the server is listening on
        GSList* uris = soup_server_get_uris(m_pSoupServer);
        for(GSList* uri = uris; uri; uri = uri->next) 
        {
            char* uriStr = soup_uri_to_string((SoupURI*)uri->data, 0);
            LOG_INFO("Soup Server listing on " << uriStr);
            g_free(uriStr);
            soup_uri_free((SoupURI*)uri->data);
        }
        g_slist_free(uris);    

        g_mutex_init(&m_serverMutex);
    }

    SoupServerMgr::~SoupServerMgr()
    {
        LOG_FUNC();

        {
            LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_serverMutex);
            if (m_pSoupServer)
            {
                g_object_unref(G_OBJECT(m_pSoupServer));        
            }

            if (m_pJsonParser != NULL)
            {
                g_object_unref(G_OBJECT(m_pJsonParser));
            }
        }
        g_mutex_clear(&m_serverMutex);

    }

    bool SoupServerMgr::AddClientBin(gpointer pClientBin)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_serverMutex);
        
        if (HasClientBin(pClientBin))
        {
            LOG_ERROR("WebRTC client-bin " << std::to_string((uint64_t)pClientBin) 
                << " is already a child of the Soup Server Manager");
            return false;
        }

        // Create a new ClientReceiver Object for the client webrtcbin
        m_clientReceivers[pClientBin] = 
            std::shared_ptr<ClientReceiver>(new ClientReceiver(pClientBin));
                        
        LOG_INFO("WebRTC client-bin "<< std::to_string((uint64_t)pClientBin) 
            << " added to the Soup Server Manager successfully");
        
        return true;
    }
    
    bool SoupServerMgr::RemoveClientBin(gpointer pClientBin)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_serverMutex);
        
        if (!HasClientBin(pClientBin))
        {
            LOG_ERROR("WebRTC client-bin " << std::to_string((uint64_t)pClientBin) 
                << " was never added the Soup Server Manager");
            return false;
        }

        // TODO - close connection if open.

        m_clientReceivers.erase(pClientBin);
                        
        LOG_INFO("WebRTC client-bin "<< std::to_string((uint64_t)pClientBin) 
            << " removed from the Soup Server Manager successfully");
        
        return true;
    }

    bool SoupServerMgr::HasClientBin(gpointer pClientBin)
    {
        LOG_FUNC();

        return (m_clientReceivers.find(pClientBin) != m_clientReceivers.end());
    }

    gpointer SoupServerMgr::GetClientBin(SoupWebsocketConnection* pConnection)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_serverMutex);

        for (auto &imap: m_clientReceivers)
        {
            if (imap.second->pConnection == pConnection)
            {
                return imap.first;
            }
        }
        // Connection was not found
        LOG_WARN("A WebRTC client-bin was not found for Connection "
            << std::to_string((uint64_t)pConnection));
        return NULL;
    }


    void SoupServerMgr::HandleOpen(SoupWebsocketConnection* pConnection)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_serverMutex);

        // Look for first unconnected client webrtcbin
        for (auto &imap: m_clientReceivers)
        {
            if (imap.second->pConnection == NULL)
            {
                // map the connection to the client
                imap.second->pConnection = pConnection;

                // connect common callbacks to Websocked "closed" and "message"
                g_signal_connect(G_OBJECT(pConnection), "closed", 
                    G_CALLBACK(on_soup_websocket_closed_cb), (gpointer)this);
                g_signal_connect(G_OBJECT(pConnection), "message", 
                    G_CALLBACK(on_soup_websocket_message_cb), (gpointer)this);
                return;
            }
        }
        LOG_ERROR("No available WebRTC client-bins available.");

    }

    void SoupServerMgr::HandleClose(SoupWebsocketConnection* pConnection)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_serverMutex);

        gpointer pClientBin = GetClientBin(pConnection);
        if (!pClientBin)
        {
            LOG_ERROR("No WebRTC client-bin for connection " 
                << std::to_string((uint64_t)pConnection));
            return;
        }
        m_clientReceivers[pClientBin] = NULL;
    }

    void SoupServerMgr::HandleMessage(SoupWebsocketConnection* pConnection, 
        SoupWebsocketDataType dataType, GBytes* message)
    {
        LOG_FUNC();
        LOCK_MUTEX_FOR_CURRENT_SCOPE(&m_serverMutex);

        // Get the client webrtcbin based on the connection
        gpointer pClientBin = GetClientBin(pConnection);
        if (!pClientBin)
        {
            LOG_ERROR("No WebRTC client-bin for connection " << std::to_string((uint64_t)pConnection));
            return;
        }

        switch (dataType)
        {
            case SOUP_WEBSOCKET_DATA_BINARY:
                LOG_ERROR("Soup Server Manager received unknown binary message, ignoring");
                g_bytes_unref(message);
                return;

            case SOUP_WEBSOCKET_DATA_TEXT:
                break;

            default:
                LOG_ERROR("Soup Server Manager received unknown data type, ignoring");
                g_bytes_unref(message);
                return;
        }

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
            LOG_ERROR("Soup Server Manager received a message a without a JSON Root");
            return;
        } 

        JsonObject* pRootJsonObject = json_node_get_object(pRootJson);
        if (!json_object_has_member(pRootJsonObject, "type")) 
        {
            LOG_ERROR("Soup Server Manager received a message without a type memeber");
            return;
        }

        const gchar* typeString = json_object_get_string_member(pRootJsonObject, "type");
        if (!json_object_has_member(pRootJsonObject, "data")) 
        {
            LOG_ERROR("Soup Server Manager received a message without data");
            return;
        }

        JsonObject* pDataJsonObject = json_object_get_object_member(pRootJsonObject, "data");
        if (g_strcmp0(typeString, "sdp") == 0) 
        {
            if (!json_object_has_member(pDataJsonObject, "type")) 
            {
                LOG_ERROR("Soup Server Manager received a SDP message without type field");
                return;
            }

            const gchar* sdpTypeString = json_object_get_string_member(pDataJsonObject, "type");
            if (g_strcmp0 (sdpTypeString, "answer") != 0) 
            {
                LOG_ERROR("Soup Server Manager expected SDP message without type 'answer' but received "
                    << sdpTypeString << "");
                return;
            }

            if (!json_object_has_member(pDataJsonObject, "sdp")) 
            {
                LOG_ERROR("Soup Server Manager received a SDP message without SDP string");
                return;
            }

            const gchar* sdpString = json_object_get_string_member (pDataJsonObject, "sdp");

            //gst_print ("Received SDP:\n%s\n", sdp_string);

            GstSDPMessage *sdp;
            if (gst_sdp_message_new(&sdp) != GST_SDP_OK)
            {
                LOG_ERROR("Soup Server Manager failed to create new SDP message");
                return;
            }

            int ret = gst_sdp_message_parse_buffer((guint8 *)sdpString, strlen(sdpString), sdp);
            if (ret != GST_SDP_OK) 
            {
                LOG_ERROR("Soup Server Manager failed to parse SDP message");
                return;
            }

            GstWebRTCSessionDescription* answer = gst_webrtc_session_description_new(
                    GST_WEBRTC_SDP_TYPE_ANSWER, sdp);
            if (!answer)
            {
                LOG_ERROR("SoupServer failed to create new webrtc session answer");
                return;
            }

            GstPromise* promise = gst_promise_new_with_change_func(on_remote_desc_set_cb, 
                pClientBin, NULL);

            g_signal_emit_by_name(G_OBJECT(pClientBin), "set-remote-description", answer, promise);    
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
                LOG_ERROR("SoupServer received ICE message without mline index");
                return;
            }

            if (!json_object_has_member(pDataJsonObject, "candidate")) 
            {
                LOG_ERROR("SoupServer received ICE message without ICE candidate string");
                return;
            }

            const gchar* candidateString = json_object_get_string_member(pDataJsonObject, "candidate");
            guint mlineIndex = json_object_get_int_member(pDataJsonObject, "sdpMLineIndex");

            LOG_INFO("Received ICE candidate with mline index: " << std::to_string(mlineIndex)
                << "; candidate: " << candidateString);

            g_signal_emit_by_name(pClientBin, "add-ice-candidate", mlineIndex, candidateString);
        }
        else
        {
            LOG_ERROR("SoupServer received unknown message type " << typeString 
                << ", returning");
        }
    }

    void SoupServerMgr::OnRemoteDescSet(GstPromise* promise)
    {
        LOG_FUNC();

        GstStructure const *reply = gst_promise_get_reply(promise);
        if (reply != NULL)
        {
            gchar* replyStr = gst_structure_to_string(reply);
            LOG_INFO("Soup Server Manager received reply for on-remote-desc-set is '" 
                << replyStr << "'");
            g_free(replyStr);
        }
        gst_promise_unref(promise);  
    }



    static void on_soup_websocket_opened_cb(G_GNUC_UNUSED SoupServer* pServer, 
        SoupWebsocketConnection* pConnection, G_GNUC_UNUSED const char *path,
        G_GNUC_UNUSED SoupClientContext* clientContext, gpointer pSoupServerMgr)
    {
        static_cast<SoupServerMgr*>(pSoupServerMgr)->HandleOpen(pConnection);
    }

    static void on_soup_websocket_closed_cb(SoupWebsocketConnection * pConnection, 
        gpointer pSoupServerMgr)
    {
        static_cast<SoupServerMgr*>(pSoupServerMgr)->HandleClose(pConnection);
    }

    static void on_soup_websocket_message_cb(SoupWebsocketConnection* pConnection, 
        SoupWebsocketDataType dataType, GBytes* message, gpointer pSoupServerMgr)
    {
        static_cast<SoupServerMgr*>(pSoupServerMgr)->HandleMessage(pConnection,
            dataType, message);
    }

    static void on_remote_desc_set_cb(GstPromise * promise, gpointer pSoupServerMgr)
    {
        static_cast<SoupServerMgr*>(pSoupServerMgr)->OnRemoteDescSet(promise);
    }

}